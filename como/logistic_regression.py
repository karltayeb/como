from abc import ABC, abstractmethod
from tkinter import W
from jax import jit
import jax.scipy as jsp
import jax.numpy as jnp
from .logistic_susie import susie_iter, Xb_susie, elbo_susie, init_susie, update_xi_susie
from .utils import get_credible_set

class LogisticRegression(ABC):
    def __init__(self, data):
        self.data = data
        self.params = {}
        self.hypers = {}

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evidence(self):
        pass

class LogisticSusie(LogisticRegression):
    def __init__(self, data, L=10):
        self.data = data
        self.params, self.hypers = init_susie(data, L)
        self.params.update(update_xi_susie(self.data, self.params))
        self.L = L

    def predict(self, X=None):
        if X is None:
            X = self.data['X']
        return Xb_susie(self.data, self.params)

    def update(self):
        self.params, self.hypers = susie_iter(
            self.data, self.params, self.hypers)

    def evidence(self):
        return elbo_susie(self.data, self.params, self.hypers)

    def report_credible_sets(self, coverage=0.95):
        return {
            f'L{k+1}': get_credible_set(self.params['alpha'][k], coverage)
            for k in range(self.L)
        }