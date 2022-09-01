import jax.numpy as jnp
import numpy as np
import jax

def sigmoid(x):
    return jax.nn.sigmoid(x)


def lamb(xi):
    return 0.5/xi * (sigmoid(xi) - 0.5)


def expected_log_logistic_bound(mu, var, xi):
    return jnp.log(sigmoid(xi)) + 0.5 * (mu - xi) + lamb(xi) * (var + mu**2 - xi^2)


def polya_gamma_mean(b, c):
    """
    Mean of PG(b, c)
    """
    # TODO: PG(b, 0) = b/4, deal with divide by 0
    return 0.5 * b/c * jnp.tanh(c/2)


# KL Divergences
def categorical_kl(alpha, pi):
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))


def normal_kl(mu, var, mu0=0, var0=1):
    return 0.5 * (jnp.log(var0) - jnp.log(var) + var/var0 + (mu - mu0)**2/var0 - 1)


def polya_gamma_kl(b, c):
    """
    Compute KL(PG(b, c) || PG(b, 0))
    Note that while we can't write the density in closed form
    the density ratio is available due to exponential tilting
    """
    # if jnp.isclose(c, 0):
    #     kl = 0
    # else:
    kl = -0.5 * c**2 * polya_gamma_mean(b, c) + b * jnp.log(jnp.cosh(c/2))
    return kl

# entropy
def bernoulli_entropy(p):
    p = jnp.minimum(p, 1-p) 
    p = jax.lax.clamp(1e-10, p, 0.5)
    q = 1 - p
    return - p * jnp.log(p) - q * jnp.log(q)


def categorical_entropy(pi):
    return -jnp.sum(pi * jnp.log(pi))


# vectorized over fist axis
categorical_entropy_vec = jax.vmap(categorical_entropy, 0, 0)


def get_credible_set(alpha, target_coverage=0.95):
    u = alpha.argsort()[::-1]
    alpha_tot = jnp.cumsum(alpha[u])[::-1]
    idx = sum(alpha_tot >= target_coverage) - 1
    cs = u[:-idx][::-1]
    return cs


# check if array is monotone increasing
is_monotone = lambda x: np.alltrue((np.roll(x, -1) - x)[:-1] >=0)


# get smallest difference between two adjacent values in an array:w
min_delta = lambda x: np.min((np.roll(x, -1) - x)[:-1])