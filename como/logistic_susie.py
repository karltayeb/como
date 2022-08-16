from re import I
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import jax
from jax import jit

from .logistic_ser import expected_beta_ser, Xb_ser, iter_ser, ser_kl, lamb, _compute_tau
from .utils import sigmoid

# compute KL for each SER, vectorized
v_ser_kl = jax.vmap(ser_kl, (
    {'alpha': 0, 'delta': 0, 'mu': 0, 'var': 0, 'tau': None, 'xi': None},
    {'pi': 0, 'sigma0': 0}))

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

def calc_Q(data, params):
    B = params['mu'] * params['alpha']
    XB = data['X'] @ B.T
    Xb = XB.sum(1)
    Zd = data['Z'] @ params['delta'].sum(0)
    B2 = params['alpha'] * (params['mu']**2 + params['var'])
    Xb2= data['X']**2 @ B2.sum(0) + Xb**2 - (XB**2).sum(1)
    Q = Xb2 + 2*Xb*Zd + Zd**2
    return Q

# TODO: impliment stochast versions for speed
# TODO: sparse matmul for speed
def Xb_stoch(data, params):
    pass
def calc_Q_stoch(data, params):
    pass

def loglik_susie(data, params):
    '''
    Compute expected log likelihood E[lnp(y | X, Z, beta, delta)]
    per data point
    '''
    Xb = Xb_susie(data, params)
    #Q = calc_Q(data, params)
    res = jnp.log(sigmoid(params['xi'])) + \
        (data['y'] - 0.5) * Xb + \
        -0.5 * params['xi'] + \
        0 # -lamb(params['xi']) * (Q - params['xi']**2)
    return(res)

@jit
def elbo_susie(data, params, hypers):
    return jnp.sum(loglik_susie(data, params)) - \
            susie_kl(params, hypers)

def update_xi_susie(data, params):
    Q = calc_Q(data, params)
    xi = jnp.sqrt(jnp.abs(Q))
    tau = _compute_tau(data, xi)
    return dict(xi=xi, tau=tau)
  
def init_susie(data, L=10):
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
        'pi': jnp.ones((L, p))/p
    }
    return params, hypers

def susie_update_ser(carry, val):
    """
    Parameters
        carry: (data, offset, xi, tau) tuple
        val: (params, hypers) the set of 
    """
    # unpack
    data, offset, xi, tau = carry
    params, hypers = val

    offset = offset - Xb_ser(data, params, jnp.zeros_like(offset))
    params['xi'] = xi
    params['tau'] = tau
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
    offset = offset + Xb_ser(data, params, jnp.zeros_like(offset))

    # offset = offset + Xb_ser(data, params, hypers)
    return (data, offset, xi, tau), (params, hypers)

@jit
def susie_iter(data, params, hypers):
    # update SERs

    # take xi and tau out of params while we update beta
    xi = params.pop('xi')
    tau = params.pop('tau')
    offset = Xb_susie(data, params)

    # package data and parameters to be updated
    carry = (data, offset, xi, tau)
    val = (params, hypers)

    # scan over parameters for each single effect
    carry, val = jax.lax.scan(susie_update_ser, carry, val)

    # update xi
    params, hypers = val
    params.update(update_xi_susie(data, params))
    return params, hypers

def f_iter(val):
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
    params, hypers = init_susie(data, L)
    params.update(update_xi_susie(data, params))

    elbo = jnp.zeros(niter)
    n, p = data['X'].shape
    
    init = (data, params, hypers, elbo, 1e6, 0)
    _, params, hypers, elbo, diff, iter = jax.lax.while_loop(
        lambda v: (v[-1] < niter) & (v[-2] > tol) , f_iter, init)

    # params, hypers, elbo, iter = jax.lax.fori_loop(0, niter, f_iter, init)
    elbo = elbo[:iter][::-1]
    return params, hypers, elbo, diff, iter