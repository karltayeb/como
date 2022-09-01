from abc import ABC, abstractmethod
from tkinter import W
from jax import jit
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
from .logistic_ser import Xb2_ser, Xb_ser, elbo_ser, iter_ser, ser_kl, pg_kl, init_ser, update_xi_ser
from .logistic_susie import susie_iter, Xb_susie, Xb2_susie, elbo_susie, init_susie, update_xi_susie, loglik_susie, susie_kl
from .utils import get_credible_set

class LogisticRegression(ABC):
    """
    Abstract method for logistic regression
    must impliment `predict`, `evidence`, and `update`
    """
    def __init__(self, data):
        self.data = data
        self.params = {}
        self.hypers = {}
        self.frozen = False

    @abstractmethod
    def predict(self):
        """
        predict on log-odds scale
        """
        pass

    @abstractmethod
    def evidence(self):
        """
        compute log likelihood/ELBO 
        """
        pass

    @abstractmethod
    def update(self):
        """
        update parameters of the logistic regression
        """
        pass

    def freeze(self):
        """
        option to freeze the distribution-- 
        do not update when update() is called
        """
        self.frozen = True

    def thaw(self):
        """
        unfreeze allows update to modify the object
        """
        self.frozen = False

class LogisticSER(LogisticRegression):
    """
    Logistic SuSiE, inherits LogisticRegression
    model stores (variational) parameters and data for logistic SuSiE
    and impliments coordinate ascent updates
    Parameters:
        data: a dictionary with keys 'y', 'X', and 'Z' for binary response, annotations, and fixed covariates resp.
        L: number of single effects
        idx: the index, if we are holding information for multiple objects in data
    """
    def __init__(self, data, idx=None):
        super().__init__(data)
        self.params, self.hypers = init_ser(data)
        self.params.update(update_xi_ser(self.data, self.params, self.hypers, 0.))

    def predict(self, X=None):
        """
        Compute expected predictions E[Xb]
        """
        if X is None:
            X = self.data['X']
        return Xb_ser(self.data, self.params)

    def predict2(self, X=None):
        """
        Compute expected predictions E[Xb]
        """
        if X is None:
            X = self.data['X']
        return Xb2_ser(self.data, self.params)

    def update(self):
        """
        Update parameters
        """
        if not self.frozen:
            self.params, self.hypers = iter_ser(
                self.data, self.params, self.hypers, 0.)

    def evidence(self):
        """
        Compute ELBO
        """
        # TODO: allow use of offset?
        return elbo_ser(self.data, self.params, self.hypers, offset=0.)
    
    def divergence(self):
        """
        """
        kl = ser_kl(self.params, self.hypers) + pg_kl(self.data, self.params, self.hypers)
        return kl

    def report_credible_sets(self, coverage=0.95):
        """
        report credible set with target coverage `coverage`
        """
        return {
            f'CS': get_credible_set(self.params['alpha'], coverage)
        }
    
    @property
    def intercept(self):
        """
        Point estimate of intercept is sum of intercepts for each SER
        """
        return self.params['delta'][:,0 ].sum()

class LogisticSusie(LogisticRegression):
    """
    Logistic SuSiE, inherits LogisticRegression
    model stores (variational) parameters and data for logistic SuSiE
    and impliments coordinate ascent updates
    Parameters:
        data: a dictionary with keys 'y', 'X', and 'Z' for binary response, annotations, and fixed covariates resp.
        L: number of single effects
        idx: the index, if we are holding information for multiple objects in data
    """
    def __init__(self, data, L=10, idx=None):
        super().__init__(data)
        self.params, self.hypers = init_susie(data, L, idx)
        self.params.update(update_xi_susie(self.data, self.params, self.hypers))
        self.L = L

    def predict(self, X=None):
        """
        Compute expected predictions E[Xb]
        """
        if X is None:
            X = self.data['X']
        return Xb_susie(self.data, self.params)

    def predict2(self, X=None):
        """
        Compute expected predictions E[Xb]
        """
        if X is None:
            X = self.data['X']
        return Xb2_susie(self.data, self.params)

    def update(self):
        """
        Update parameters
        """
        if not self.frozen:
            self.params, self.hypers = susie_iter(
                self.data, self.params, self.hypers)

    def evidence(self):
        """
        Compute ELBO
        """
        return elbo_susie(self.data, self.params, self.hypers)
    
    def divergence(self):
        """
        """
        kl = susie_kl(self.params, self.hypers) + pg_kl(self.data, self.params, self.hypers)
        return kl

    def report_credible_sets(self, coverage=0.95):
        """
        report credible set with target coverage `coverage`
        """
        return {
            f'L{k+1}': get_credible_set(self.params['alpha'][k], coverage)
            for k in range(self.L)
        }
    
    @property
    def intercept(self):
        """
        Point estimate of intercept is sum of intercepts for each SER
        """
        return self.params['delta'][:,0 ].sum()

class InterceptOnly(LogisticRegression):
    """
    Simplest case where we ignore the covariates and just estimate a global \pi_1
    """
    def __init__(self, data):
        super().__init__(data)
        self.intercept = 0

    def update(self):
        if not self.frozen:
            ybar = np.mean(self.data['y'])
            self.intercept = np.log(ybar) - np.log(1 - ybar)

    def evidence(self):
        """
        binomial log-likelihood under the current intercept
        """
        y = self.data['y']
        return np.sum(y) * self.intercept \
            + y.size * (np.log(1) - np.log(1 + np.exp(self.intercept)))

    def predict(self):
        """
        InterceptOnly predicts the intercept for all observations
        """
        return self.intercept

