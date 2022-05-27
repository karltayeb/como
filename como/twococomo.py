from termios import FF0
from tkinter import W
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import jax

from .component_distributions import ComponentDistribution
from .logistic_regression import LogisticRegression

class TwoComponentCoMo:
    def __init__(self, data, f0: ComponentDistribution, f1: ComponentDistribution, logreg: LogisticRegression):
        """
        Initialize Two component covariate moderated EBNM
    
        Parameters:
            data: dictionary with keys 'beta' and 'se' for observations and standard errors
            f0: a ComponentDistribution object for the null distribtuion
            f1: a ComponentDistribution object for the active distribtuion
            logreg: a LogisticRegression model (e.g. LogisticSusie)
        """
        self.data = data
        self.f0 = f0
        self.f1 = f1
        self.logreg = logreg
        self.responsibilities = twococomo_compute_responsibilities(
            self.data, self.logreg, self.f0, self.f1
        )
    
    def loglik(self):
        return twococomo_loglik(
            self.data, self.responsibilities, self.f0, self.f1 
        )

    def elbo(self):
        return twococomo_elbo(
            self.data, self.responsibilities, self.f0, self.f1, self.logreg  
        )
    
    def update_responsibilities(self):
        """
        update the responsibilities, notify the logistic regression too
        """
        self.responsibilities = twococomo_compute_responsibilities(
            self.data, self.logreg, self.f0, self.f1)
        self.logreg.data['y'] = self.responsibilities

    def iter(self):
        self.update_responsibilities()
        self.logreg.update()
        self.f0.update()
        self.f1.update()


# The two component covariate moderated EBNM
def twococomo_compute_responsibilities(data, logreg, f0, f1):
    """
    compute p(\gamma | data, f0, f1)
    """
    logit_pi = logreg.predict()
    f0_loglik = f0.convolved_logpdf(data['beta'], data['se'])
    f1_loglik = f1.convolved_logpdf(data['beta'], data['se'])

    logits = f1_loglik - f0_loglik + logit_pi
    responsibilities = 1/(1 + jnp.exp(-logits))
    return responsibilities

def twococomo_loglik(data, responsibilities, f0, f1):
    """
    compute E[p(\beta | s, \gamma, f_0, f_1)]
    """
    f0_loglik = f0.convolved_logpdf(data['beta'], data['se'])
    f1_loglik = f1.convolved_logpdf(data['beta'], data['se'])
    loglik = jnp.sum((1- responsibilities) * f0_loglik + responsibilities * f1_loglik)
    return loglik

def twococomo_elbo(data, responsibilities, f0, f1, logreg):
    data_loglik = twococomo_loglik(data, responsibilities, f0, f1)
    assignment_entropy =  -jnp.sum(responsibilities * jnp.log(responsibilities))
    logreg_elbo = logreg.evidence()
    total_elbo = data_loglik + assignment_entropy + logreg_elbo
    return dict(
        data_loglik=data_loglik,
        assignment_entropy=assignment_entropy,
        logistic_elbo=logreg_elbo,
        total_elbo=total_elbo)

def twococomo_iter(data, f0, f1, logreg):
    # update responsibilities, stored as ['y']
    res = twococomo_compute_responsibilities(data, logreg, f0, f1)
    logreg.data['y'] = res
    logreg.update()
    f0.update()
    f1.update()








