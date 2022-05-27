from abc import ABC, abstractmethod
from jax import jit
import jax.scipy as jsp
import jax.numpy as jnp

class ComponentDistribution(ABC):
    @abstractmethod
    def convolved_logpdf(self, beta, se, sigma0):
        pass


class PointMassComponent(ComponentDistribution):
    def __init__(self, loc):
        self.loc = loc

    def convolved_logpdf(self, beta, se):
        return _normal_convolved_logpdf(
            beta, se, self.loc, 0)
    def update(self):
        pass

class NormalComponent(ComponentDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def convolved_logpdf(self, beta, se):
        return _normal_convolved_logpdf(
            beta, se, self.loc, self.scale)

    def update(self):
        pass

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

