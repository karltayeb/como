import jax.numpy as jnp
import numpy as np
import jax

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def lamb(xi):
    return 0.5/xi * (sigmoid(xi) - 0.5)

def expected_log_logistic_bound(mu, var, xi):
    return jnp.log(sigmoid(xi)) + 0.5 * (mu - xi) + lamb(xi) * (var + mu**2 - xi^2)

# KL Divergences
def categorical_kl(alpha, pi):
    return jnp.nansum(alpha * (jnp.log(alpha) - jnp.log(pi)))

def normal_kl(mu, var, mu0=0, var0=1):
    return 0.5 * (jnp.log(var0) - jnp.log(var) + var/var0 + (mu - mu0)**2/var0 - 1)

def bernoulli_entropy(p):
    p = jnp.minimum(p, 1-p) 
    p = jax.lax.clamp(1e-10, p, 0.5)
    q = 1 - p
    return - p * jnp.log(p) - q * jnp.log(q)

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