from operator import length_hint
from tkinter import W
from typing import List

from .component_distributions import ComponentDistribution, eta2pi
from .logistic_regression import LogisticRegression
from .utils import categorical_entropy_vec, polya_gamma_kl

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit

class MoreComponentCoMo:
    def __init__(self, data, f_list: List[ComponentDistribution], logreg_list: List[LogisticRegression]):
        """
        Initialize more component covariate moderated EBNM, with K components (inferred from f_list)
    
        Parameters:
            data: dictionary with keys 'beta' and 'se' for observations and standard errors
            f_list: a list of ComponentDistributions, length K
            logreg_list: a list of Logistic Regression models, length K-1
        """
        self.data = data
        self.f_list = f_list
        self.logreg_list = logreg_list

        # initialize data for logistic regressions
        self.data['Y'] = mococomo_compute_responsibilities(
            self.data, self.f_list, self.logreg_list
        )
        self.data['N'] = 1. - (jnp.cumsum(self.data['Y'], 1) - self.data['Y'])

        self.elbo_history = []

    @property
    def responsibilities(self):
        """
        posterior assignment probability N x K matrix
        """
        return np.array(self.data['Y'])

    @property
    def prior_assigment_probability(self):
        """
        return the (expected) prior log odds for each observation
        this is the (expectation of) the prediction from the logistic regression model
        """
        return mococomo_prior_mixture_weights(self.logreg_list)

    @property
    def beta(self):
        return self.data['beta']
    
    @property
    def se(self):
        return self.data['se']
    
    @property
    def X(self):
        return self.data['X']

    @property
    def Z(self):
        return self.data['Z']

    @property
    def post_mean(self):
        """
        posterior mean of two component mixture
        """
        # TODO: compute mixture mean
        pass
        
    @property
    def post_mean2(self):
        # compute mixture 2nd moment
        pass

    @property
    def post_var(self):
        """
        posterior variance
        """ 
        # compute mixture mean
        pass

    def loglik(self):
        return mococomo_loglik(
            self.data, self.f_list, self.responsibilities 
        )

    def elbo(self, record: bool = False):
        new_elbo = mococomo_elbo(
            self.data, self.f_list, self.logreg_list, self.responsibilities
        )
        if record:
            self.elbo_history.append(new_elbo['total_elbo'])
        return new_elbo
    
    def update_responsibilities(self):
        """
        update the responsibilities and pass values to logreg
        since this is MoCoCoMo, we need to also record cumulative probabilities
        """
        self.data['Y'] = mococomo_compute_responsibilities(
            self.data, self.f_list, self.logreg_list
        )
        self.data['N'] = 1. - (jnp.cumsum(self.data['Y'], 1) - self.data['Y'])

    def update_logreg(self):
        """
        Update the logistic regression
        """
        [lr.update() for lr in self.logreg_list];

    def update_f(self):
        """
        Update the component distributions
        """
        [f.update(self.data) for f in self.f_list];

    def iter(self):
        """
        iteration updates responsibilities, logistic regression, f0, and f1
        """
        self.update_responsibilities()
        self.update_logreg()
        self.update_f()

    def fit(self, niter=100, tol=1e-3):
        for i in range(niter):
            self.iter()
            self.elbo_history.append(self.elbo()['total_elbo'])
            if self.converged(tol):
                break

    def converged(self, tol=1e-3):
        return ((len(self.elbo_history) > 2)
            and ((self.elbo_history[-1] - self.elbo_history[-2]) < tol))


###
# Convert pi_tilde to pi-- prior mixture weights
###
def pi_scanner(gamma, pi_tilde):
    """
    produce the next pi from the current
    cumulative probability and conditional probability
    use this function to scan over conditional probabilities pi_tilde
    """
    pi = pi_tilde * (1. - gamma)
    gamma = gamma + pi
    return gamma, pi


def pi_tilde2pi(pi_tilde):
    """
    convert vector of conditional mixture probabilities
    to mixture probabilities pi

    pi_tilde[k] = P(draw k given we did not draw 1... k-1)
    pi[k] = P(draw k)
    """
    # append conditional probability of last state
    pi_tilde = jnp.concatenate([pi_tilde, jnp.array([1.0])])
    _, pi = jax.lax.scan(pi_scanner, init=0., xs = pi_tilde)
    return pi


# vectorized version of pi_tilde2pi
pi_tilde2pi_vec = jax.vmap(pi_tilde2pi, 0, 0)


def mococomo_prior_mixture_weights(logreg_list: List[LogisticRegression]):
    """
    compute prior on mixture assignments from covariates
    """
    # K-1 x N with the last row all ones
    pi_tilde = jnp.array([jax.nn.sigmoid(lr.predict()) for lr in logreg_list]).T

    # K x N
    pi = pi_tilde2pi_vec(pi_tilde)
    return pi


_eta2pi_vec = jax.vmap(eta2pi, 0, 0)

from .logistic_susie import Xb2_susie
from .logistic_ser import E_omega

def mococomo_logpi(logreg_list: List[LogisticRegression]):
    """
    compute log(pi(x)) up to a constant
    Xb_k - 0.5 \sum_{j < k} Xb_j - kln2 + c
    adjusted so that logpi_K = 0
    """
    K = len(logreg_list) + 1
    Xb = jnp.array([f.predict() for f in logreg_list]).T  # K-1
    kln2 = (jnp.arange(K - 1) + 1) * jnp.log(2)[None]

    Xb2 = jnp.array([Xb2_susie(f.data, f.params) for f in logreg_list]).T
    omega = jnp.array([E_omega(f.data, f.params, f.hypers) for f in logreg_list]).T
    D = jnp.cumsum(Xb2 * omega, axis=1)

    C = jnp.log(2) * (K-1) + 0.5 * jnp.sum(Xb, axis=1)[:, None] + 0.5 * D
    logpi = Xb - 0.5 * jnp.cumsum(Xb, axis=1) - 0.5 * D - kln2 + C
    return jnp.log(_eta2pi_vec(logpi))

###
# Compute posterior assignment probabilities
###

def mococomo_compute_responsibilities(data: dict, f_list: List[ComponentDistribution], logreg_list: List[LogisticRegression]):
    """
    get posterior assignment probabilities
    """
    # N x K log likelihood of data under each component distribution
    loglik = jnp.array([
        f.convolved_logpdf(data['beta'], data['se']) for f in f_list]).T
    #logpi = jnp.log(mococomo_prior_mixture_weights(logreg_list))
    logpi = mococomo_logpi(logreg_list)
    post_pi = jax.nn.softmax(loglik + logpi, axis=1)
    return post_pi


def mococomo_loglik(data: dict, f_list: List[ComponentDistribution], sum: bool=True):
    """
    Compute the expected data log likelihood under mococomo model E_q(z)[logp(y | z)]
    Parameters:
        data: a dictionary of data
        f_list: a list of Component distribution objects K
        sum: Boolean, if True sum loglik across observations, 
            if False return loglikelihood for each observations
    """
    loglik = jnp.array([
        f.convolved_logpdf(data['beta'], data['se']) for f in f_list]).T
    pi = data['Y']  # posterior assignment probabilities

    if sum:  # one number, data log likelihood
        loglik = jnp.sum(loglik + pi)
    else:  # log likelihood of each observation seperately
        loglik = jnp.sum(loglik + pi, 1)
    return loglik

def mococomo_elbo(data, f_list, logreg_list):
    pi = data['Y']
    data_loglik = jnp.sum(mococomo_loglik(data, f_list, sum = False))
    assignment_entropy = jnp.sum(categorical_entropy_vec(pi))
    logreg_elbo = sum([ll.evidence() for ll in logreg_list])
    total_elbo = data_loglik + assignment_entropy + logreg_elbo
    
    return dict(
        data_loglik=data_loglik,
        assignment_entropy=assignment_entropy,
        logreg_elbo = logreg_elbo,
        total_elbo=total_elbo)

from .logistic_susie import Xb_susie, Xb2_susie
def mococomo_assignment_loglik(logreg_list, responsibilities):
    """
    compute E_q(beta, omega)[p(z | beta, omega)]
    """
    Xb = jnp.array([Xb_susie(f.data, f.params) for f in logreg_list]).T
    Xb2 = jnp.array([Xb2_susie(f.data, f.params) for f in logreg_list]).T 
    omega = jnp.array([E_omega(f.data, f.params, f.hypers) for f in logreg_list]).T

    A = jnp.cumsum(Xb, axis=1)
    B = - 0.5 * jnp.sum(Xb2 * omega, axis=1)

    K = len(logreg_list) + 1
    kln2 = (jnp.arange(K-1) * jnp.log(2))[None]

    loglik = jnp.sum((-kln2 + Xb - 0.5 * A) * responsibilities[:, :-1], axis=1)
    loglik_K = -(K-1) * jnp.log(2) - 0.5 * A[:, -1] * responsibilities[:, -1]
    loglik = loglik + loglik_K + B
    return loglik
    
def mococomo_elbo2(data, f_list, logreg_list, responsibilities):
    data_loglik = jnp.sum(mococomo_loglik(data, f_list, responsibilities, sum = False))
    assignment_entropy = jnp.sum(categorical_entropy_vec(responsibilities))
    assignment_likelihood = jnp.sum(mococomo_assignment_loglik(logreg_list, responsibilities))

    susie_kl = sum([ll.divergence() for ll in logreg_list])
    
    total_elbo = data_loglik + assignment_likelihood + assignment_entropy - susie_kl
    return dict(
        data_loglik = data_loglik,
        assignment_entropy = assignment_entropy,
        assignment_likelihood = assignment_likelihood,
        susie_kl = susie_kl,
        total_elbo = total_elbo
    )


def comp_Nk(responsiblities):
    Nk = 1 - jnp.cumsum(responsiblities)
    return Nk