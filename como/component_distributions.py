from abc import ABC, abstractmethod
from re import I
from tkinter import W
from jax import jit, vmap, grad, hessian
import jax
import jax.scipy as jsp
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import numpy as np

class ComponentDistribution(ABC):
    def __init__(self):
        self.frozen = False

    @abstractmethod
    def convolved_logpdf(self, beta, se, sigma0):
        pass

    @abstractmethod
    def update(self, data):
        """
        update the parameters for the component distribution 
        Parameters:
            data: a dictionary with data
        """
        pass

    def freeze(self):
        self.frozen = True

    def thaw(self):
        self.frozen = False



class PointMassComponent(ComponentDistribution):
    def __init__(self, loc: float = 0.):
        super().__init__()
        self.loc = loc

    def convolved_logpdf(self, beta, se):
        return _normal_convolved_logpdf(
            beta, se, self.loc, 0)

    def update(self, data):
        pass

class NormalFixedLocComponent(ComponentDistribution):
    def __init__(self, loc=0., scale=1.):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.scale_grid = np.linspace(0.1, 100, 1000)  # fixed grid

    def convolved_logpdf(self, beta, se):
        return _normal_convolved_logpdf(
            beta, se, self.loc, self.scale)

    def update(self, data):
        # update by picking mle on a fixed grid
        # this is actually quite fast
        if self.frozen:
            return None

        self.scale = grid_optimizie_normal_ebnm(
            beta=data['beta'], 
            se=data['se'],
            loc=self.loc,
            scale_grid=self.scale_grid,
            responsibilities=data['y']
        )
        

class NormalScaleMixtureComponent(ComponentDistribution):
    def __init__(self, loc: float = 0., scales: np.ndarray = None, pi: np.ndarray = None):
        self.loc = loc

        # TODO: pick sensible default for scale mixture-- what does ASH do?
        if scales is None:
            scales = np.power(2, np.arange(10)-5.)
        self.scales = jnp.array(scales)

        # record natural parameters for categorical
        # log(pi_k/pi_K) k= 1,..., K-1
        if pi is None:
            pi = np.ones(scales.size) / scales.size
        self.eta = pi2eta(pi)


    def convolved_logpdf(self, beta, se):
        return _nsm_convolved_logpdf(
            beta, se, self.loc, self.scales, self.eta
        )

    def update(self, data: dict, niter: int = 100):
        """
        Update mixture weights via EM (default niter=100)
        
        Note: Due to overhead, it's not that much more time-intensive to
        do 100 iterations of EM vs 1 even when you have 100k observatiosn
        """
        L = lambda i, eta: emNSM(
            data['beta'], data['se'], self.loc, self.scales, eta, data['y'])
        self.eta = jax.lax.fori_loop(0, niter, L, self.eta)

    @property
    def pi(self):
        return eta2pi(self.eta) 

"""
Normal distribution helpers
"""
@jit
def _normal_convolved_logpdf(beta, se, loc, scale):
    scale = jnp.sqrt(se**2 + scale**2)
    return jsp.stats.norm.logpdf(beta-loc, scale=scale)


def _normal_ebnm_objective(beta, se, loc, scale, responsibilities):
    """
    sum of log likelihoods for each data point, weighted by the probability of being non-null
    optimize this for component distribution update
    """
    loglik = _normal_convolved_logpdf(beta, se, loc, scale)
    return jnp.sum(responsibilities * loglik)

v_nebnm = vmap(_normal_ebnm_objective, (None, None, None, 0, None), 0)

@jit
def grid_optimizie_normal_ebnm(beta, se, loc, scale_grid, responsibilities):
  idx = jnp.argmax(v_nebnm(beta, se, loc, scale_grid, responsibilities))
  return scale_grid[idx] 


"""
Scale mixture of normal helpers
"""

def pi2eta(pi):
    eta = jnp.log(pi)
    eta = eta[:-1] - eta[-1]
    return eta


def eta2pi(eta):
    """
    map natural parameters to mixture weights
    just softmax but add a 0 for the last component
    """
    eta0 = jnp.concatenate([eta, jnp.array([0.])])
    return jax.nn.softmax(eta0)


# vectorized over scales pdf for mixture computation
_vec_normal_convolved_logpdf = vmap(
    _normal_convolved_logpdf,
    in_axes=(None, None, None, 0), out_axes=1
)


def _nsm_convolved_logpdf(beta, se, loc, scales, eta):
    """
    pdf for observations beta with standard errors se, with effects
    drawn from a mixture of normals
    """
    normal_grid = _vec_normal_convolved_logpdf(beta, se, loc, scales)
    eta0 = jnp.concatenate([eta, jnp.array([0.])])
    c = logsumexp(eta0)
    logpdf = (logsumexp(normal_grid + eta0[None], axis=1) - c)
    return logpdf

# NOTE: written in terms of unconstrained natural parameters
def lossNSM(beta, se, loc, scales, eta):
     return jnp.sum(_nsm_convolved_logpdf(
         beta, se, loc, scales, eta))
gradNSM = grad(lossNSM, argnums=4)
hessNSM = hessian(lossNSM, argnums=4)


# NOTE: should be able to do this without explicitly solving
# "natural graident" descent in exponential families
def newtonNSM(beta, se, loc, scales, eta):
    """
    natural gradient update
    """
    ng = jnp.linalg.solve(
        hessNSM(beta, se, loc, scales, eta), 
        gradNSM(beta, se, loc, scales, eta)
    )
    eta = eta - ng
    return eta


"""
EM updates for scale mixture of normals
will converge to the global optimum since objective is convex
"""
def mix_assignment_prop(beta, se, loc, scales, eta):
    """
    compute assignment probabilities
    """
    sigmas = jnp.sqrt(se**2 + scales**2)
    # eta0 = jnp.concatenate([eta, jnp.array([0.])])
    logpi = jnp.log(eta2pi(eta))
    loglik = jsp.stats.norm.logpdf(beta, loc, scale=sigmas)
    return jax.nn.softmax(loglik + logpi)

# vectorize to do over many data points
mix_assigment_prop_vec = vmap(
    mix_assignment_prop,
    (0, 0, None, None, None), 0
)

@jit
def emNSM(beta, se, loc, scales, eta, responsibilities = 1.0):
    """
    update mixture weights (average of assignment probabilities)
    """
    R = mix_assigment_prop_vec(beta, se, loc, scales, eta)
    Rsum = (responsibilities[:, None] * R).sum(0)
    pi_new = Rsum / Rsum.sum()
    return pi2eta(pi_new)
    