from abc import ABC, abstractmethod
from jax import jit, vmap
import jax.scipy as jsp
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
    def __init__(self, loc):
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
        

class UnimodalNormalMixtureComponent(ComponentDistribution):
    def __init__(self, mu, sigma0_grid, pi):
        self.mu = mu
        self.sigma0_grid = jnp.array(sigma0_grid)
        self.pi = jnp.ones_like(sigma0_grid) / sigma0_grid.size

    def convolved_logpdf(self, beta, se):
        return _normal_convolved_logpdf(
            self.mu, beta, se, sigma0)

    def update(self):
        pass

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
