from re import I
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import jax
from jax import jit

from .logistic_ser import expected_beta_ser, Xb_ser, iter_ser, ser_kl, _compute_tau, _compute_kappa, _get_y, _get_N
from .utils import polya_gamma_mean, polya_gamma_kl, sigmoid

# compute KL for each SER, vectorized
v_ser_kl = jax.vmap(ser_kl, (
    {'alpha': 0, 'delta': 0, 'mu': 0, 'var': 0, 'tau': None, 'xi': None},
    {'pi': 0, 'sigma0': 0, 'idx': None}))

def susie_kl(params, hypers):
    """
    KL divergence from the current variational approximation to the prior
    Just the sum of KLs for each SER
    """
    return jnp.sum(v_ser_kl(params, hypers))

def Ew_susie(params):
    b = (params['alpha'] * params['mu']).sum(1)
    return b

def Eb_susie(params):
    b = (params['alpha'] * params['mu']).sum(0)
    return b
    
def Xb_susie(data, params):
    '''Computes E[X \beta + Z \delta + offset]'''
    b = jnp.sum(params['alpha'] * params['mu'], 0)
    Xb = data['X'] @ b
    delta = jnp.sum(params['delta'], 0)
    Zd = data['Z'] @ delta
    pred = Xb + Zd 
    return(pred)

def Xb2_susie(data, params):
    B = params['mu'] * params['alpha']
    XB = data['X'] @ B.T
    Xb = XB.sum(1)
    Zd = data['Z'] @ params['delta'].sum(0)
    B2 = params['alpha'] * (params['mu']**2 + params['var'])
    Xb2= data['X']**2 @ B2.sum(0) + Xb**2 - (XB**2).sum(1)
    Xb2 = Xb2 + 2*Xb*Zd + Zd**2
    return Xb2 

def loglik_susie(data, params, hypers):
    '''
    Compute expected log likelihood E[lnp(y | X, Z, beta, delta)]
    per data point
    '''
    Xb = Xb_susie(data, params)
    Xb2 = params['xi']**2  # alt: Xb2_susie(data, params)
    kappa = _compute_kappa(data, hypers)
    omega = polya_gamma_mean(_get_N(data, hypers), params['xi'])
    loglik = kappa * Xb - 0.5 * omega * Xb2
    return loglik
    #Xb2 = Xb2_susie(data, params)
    # res = jnp.log(sigmoid(params['xi'])) + \
    #     (_get_y(data, hypers) - 0.5) * Xb + \
    #     -0.5 * params['xi'] + \
    #     0 # -lamb(params['xi']) * (Q - params['xi']**2)
    # return(res)

@jit
def elbo_susie(data, params, hypers):
    loglik = jnp.sum(
        loglik_susie(data, params, hypers) - polya_gamma_kl(_get_N(data, hypers), params['xi'])
    )
    kl = susie_kl(params, hypers)
    return loglik - kl

def update_xi_susie(data, params, hypers):
    Xb2 = Xb2_susie(data, params)
    xi = jnp.sqrt(jnp.abs(Xb2))
    tau = _compute_tau(data, hypers, xi)
    return dict(xi=xi, tau=tau)
  
def init_susie(data, L=10, idx=None):
    n, p = data['X'].shape
    params = {
        'mu': jnp.zeros((L, p)),
        'var': jnp.ones((L, p)),
        'alpha': jnp.ones((L, p))/p,
        'delta': jnp.zeros((L, 1)),
        'xi': jnp.zeros(n) + 1e-6,
        'tau': jnp.ones(p)
    }
    hypers = {
        'sigma0': jnp.ones(L) * 100.,
        'pi': jnp.ones((L, p))/p,
        'idx': idx 
    }
    return params, hypers

def susie_update_ser(carry, val):
    """
    Update a single SER in SuSiE, used as a subroutine in susie_iter
    Parameters
        carry: (data, offset, xi, taui, xi) tuple
        val: (params, hypers) the set of 
    """
    # unpack
    data, offset, xi, tau, idx = carry
    params, hypers = val

    # remove current SER effect
    offset = offset - Xb_ser(data, params, jnp.zeros_like(offset))

    # update the SER
    params['xi'] = xi
    params['tau'] = tau
    hypers['idx'] = idx

    params, hypers = iter_ser(
        data, params, hypers, offset,
        update_b=True,
        update_delta=True,
        update_xi=False,
        update_hypers=False,
        track_elbo=False
    )
    params.pop('xi')
    params.pop('tau')
    hypers.pop('idx')

    # add back effect of this SER
    offset = offset + Xb_ser(data, params, jnp.zeros_like(offset))
    return (data, offset, xi, tau, idx), (params, hypers)

@jit
def susie_iter(data, params, hypers):
    """
    Perform one iteration of CAVI for Binomial SuSiE
    By updating all of the SERs in sequence
    """
    # take xi and tau out of params while we update beta
    # now params and hypers only contain K x _ vectors to iterate over
    # which apparently is necessary to get jax.lax.scan to work
    xi = params.pop('xi')
    tau = params.pop('tau')
    idx = hypers.pop('idx')
    offset = Xb_susie(data, params)

    # package data and parameters to be updated
    carry = (data, offset, xi, tau, idx)
    val = (params, hypers)

    # scan over parameters for each single effect
    carry, val = jax.lax.scan(susie_update_ser, carry, val)

    # update xi, add xi and tau back to params
    params, hypers = val
    params.update(update_xi_susie(data, params, hypers))

    # add idx back to hypers
    hypers['idx'] = idx
    return params, hypers

def f_iter(val):
    """
    A single iteration of CAVI for SuSiE
    wraps susie_iter so that it can be used with jax while loop
    also records ELBO
    """
    # unpack
    data, params, hypers, elbo, diff, iter = val

    # update
    params, hypers = susie_iter(data, params, hypers)

    # book-keeping
    new_elbo = elbo_susie(data, params, hypers)
    diff = jnp.abs(new_elbo - elbo[0])
    elbo = jnp.concatenate([
        jnp.array([new_elbo]),
        elbo[:-1]
    ])
    return data, params, hypers, elbo, diff, iter+1 

def fit_susie(data, L, niter=10, tol=1e-3):
    """
    Routine for initializing and fitting SuSiE
    """
    params, hypers = init_susie(data, L)
    params.update(update_xi_susie(data, params, hypers))

    elbo = jnp.zeros(niter)
    n, p = data['X'].shape
    
    # initial state
    init_diff = 1e6
    init_iter = 0
    init = (data, params, hypers, elbo, init_diff, init_iter)

    # perform iteration until covergence, or niter reached
    _, params, hypers, elbo, diff, iter = jax.lax.while_loop(
        lambda v: (v[-1] < niter) & (v[-2] > tol) , f_iter, init)

    # clean up elbo history
    elbo = elbo[:iter][::-1]
    return params, hypers, elbo, diff, iter