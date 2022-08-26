import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import jax
from flax.core import FrozenDict

from .utils import *

# Expected beta, and other expectations
def expected_beta_ser(params):
    '''Compute E[beta] for SER'''
    b = (params['alpha'] * params['mu'])
    return b


def ser_kl(params, hypers):
    """
    Compute KL(q(\beta_l) || p(\beta_l))
    the KL divergence between the variational approximation to the posterior
    and the SER prior
    """
    return categorical_kl(params['alpha'], hypers['pi']) \
            + jnp.sum(params['alpha'] * normal_kl(
                params['mu'], params['var'], 0, hypers['sigma0']**2))


def Xb_ser(data, params, offset=0):
    '''Computes E[X \beta + Z \delta + offset]'''
    Xb = data['X'] @ expected_beta_ser(params)
    Zd = data['Z'] @ params['delta']
    pred = Xb + Zd + offset
    return(pred)


def Xb2_ser(data, params, offset):
    '''Computes E[(X \beta + Z \delta + offset)^2]'''
    Xb = data['X'] @ expected_beta_ser(params)
    Zd = data['Z'] @ params['delta'] + offset
    b2 = params['alpha'] * (params['mu']**2 + params['var'])
    Xb2 = data['X']**2 @ b2
    Q = Xb2 + 2 * Xb * Zd + Zd**2
    return Q


def jj_ser(data, params, offset = 0.):
    """
    Jaakola jordan bound
    equivalent (up to a constant?) to E[p(y | b, w) - q(w)]
    """
    xi = params['xi']
    Xb = Xb_ser(data, params, offset)
    Xb2 = Xb2_ser(data, params, offset) 
    kappa = _compute_kappa(data)
    omega = polya_gamma_mean(data.get('N', 1.), params['xi'])

    return jnp.log(jax.nn.sigmoid(xi)) \
        + kappa * Xb \
        - 0.5 * xi \
        + 0.5 * omega * (Xb2 - xi**2)


def loglik_ser(data, params, offset=0):
    """
    Compute the expected conditional data likelihood E[p(y | beta, omega)]
    y are the observed counts
    beta are the regression coefficients
    omega are the latent PG variables
    """
    Xb = Xb_ser(data, params, offset)
    Xb2 = Xb2_ser(data, params, offset)
    omega = polya_gamma_mean(data.get('N', 1.), params['xi'])
    
    # TODO: figure out how to index alpha, for when we use it in mococomo
    kappa = _compute_kappa(data)
    loglik =  kappa * Xb - 0.5 * omega * Xb2
    return loglik

def elbo_ser(data, params, hypers, offset):
    '''Compute ELBO for logistic SER'''
    loglik = jnp.sum(
        loglik_ser(data, params, offset)
        - polya_gamma_kl(data.get('N', 1.), params['xi'])
    )
    kl = ser_kl(params, hypers)
    return loglik - kl


def _compute_kappa(data):
    return data['y'] - 0.5 * data.get('N', 1.)


def _compute_nu(data, params, hypers, offset):
    """
    compute a scaled posterior mean parameter
    """
    kappa = _compute_kappa(data) 

    # when running SuSiE offset should be the residual (Xb - E[b_l])
    # when offset is 0, we do not need to compute omega
    r = offset + data['Z'] @ params['delta']
    omega = polya_gamma_mean(data.get('N', 1.), params['xi'])
    tmp = kappa - omega * r
    nu = tmp @ data['X']
    return nu


def _compute_tau(data, xi):
    """
    compute precision parameter (partial)
    only need to call once per update of xi
    """
    omega = polya_gamma_mean(data.get('N', 1.), xi)
    tau = omega @ (data['X']**2)
    return tau


def update_xi_ser(data, params, hypers, offset):
    '''update variational parameters for logistic approximation'''
    Xb2 = Xb2_ser(data, params, offset)
    xi = jnp.clip(jnp.sqrt(jnp.abs(Xb2)), 1e-20, 1e20)
    tau = _compute_tau(data, xi)
    return dict(xi=xi, tau=tau)


def update_delta_ser(data, params, hypers, offset):
    '''update intercept/covariate coefficients to maximize ELBO'''
    Z = data['Z']
    D = 2 * lamb(params['xi'])
    Xb = data['X'] @ expected_beta_ser(params) + offset
    delta = jnp.linalg.solve((D * Z.T) @ Z, Z.T @ (data['y'] - 0.5 - D * Xb))
    return delta


def update_b_ser(data, params, hypers, offset):
    '''
    Update variational parameters assocaited with SER (mu, var, alpha)
    return a dictionary with updated values
    '''
    nu = _compute_nu(data, params, hypers, offset)
    tau = (1 / hypers['sigma0']**2) + params['tau']  # only comput when we update xi
    logits = jnp.log(hypers['pi']) \
        - 0.5 * jnp.log(tau) \
        + 0.5 * nu**2/tau
    logits = logits - logsumexp(logits)
    alpha = jnp.exp(logits)
    alpha = alpha / alpha.sum()
    post = {
        'mu': nu/tau,
        'var': 1/tau,
        'alpha': alpha
    }
    return post

def update_sigma0(data, params, hypers, offset):
    """
    Variational M-step for sigma0
    Maximizes ELBO wrt point estimate sigma0
    """
    b2 = params['alpha'] * (params['mu']**2 + params['var'])
    sigma0 = jnp.sqrt(jnp.sum(b2))
    return sigma0

def init_ser(data):
    '''Initialize variational approximation for SER'''
    n, p = data['X'].shape
    params = {
        'mu': jnp.zeros(p),
        'var': jnp.ones(p),
        'alpha': jnp.ones(p)/p,
        'delta': jnp.zeros(1),
        'xi': jnp.zeros(n) + 1e-6,
        'tau': jnp.ones(p)
    }
    hypers = {
        'sigma0': 100.,
        'pi': jnp.ones(p)/p
    }
    return params, hypers

@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def iter_ser(data, params, hypers, offset, update_b=True, update_delta=True, update_xi=True, update_hypers=True, track_elbo=True):
    """
    Single iteration of SER
    """
    # update
    if update_b:
        params.update(update_b_ser(data, params, hypers, offset))
    if update_delta:
        params['delta'] = update_delta_ser(data, params, hypers, offset)
    if update_hypers:
        hypers['sigma0'] = update_sigma0(data, params, hypers, offset)
    if update_xi:
        params.update(update_xi_ser(data, params, hypers, offset))
    return params, hypers
 
def fit_ser(data, offset=0, control={}, niter=100, tol=1e-3):
    """
    Fit SER
    """
    params, hypers = init_ser(data)
    elbo = jnp.zeros(niter)

    data = FrozenDict(dict(
        X=jax.device_put(data['X']),
        y=jax.device_put(data['y']),
        Z=jax.device_put(data['Z'])
    ))
    control = FrozenDict(control)

    def f_iter(val):
        # unpack
        params, hypers, offset, elbo, diff, iter = val
        #update
        params, hypers = iter_ser(data, params, hypers, offset, **control)
        # book keeping
        if control.get('track_elbo', True):
            new_elbo = elbo_ser(data, params, hypers, offset)
            diff = jnp.abs(new_elbo - elbo[0])
            elbo = jnp.concatenate([
                jnp.array([new_elbo]),
                elbo[:-1]
            ])
        return params, hypers, offset, elbo, diff, iter+1

    init = (params, hypers, offset, elbo, 1e6, 0)
    params, hypers, offset, elbo, diff, iter = jax.lax.while_loop(
        lambda v: (v[-1] < niter) & (v[-2] > tol) , f_iter, init)
    elbo = elbo[:iter][::-1]
    return params, hypers, elbo