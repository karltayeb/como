from typing import List

from .component_distributions import ComponentDistribution
from .logistic_regression import LogisticRegression

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
        self.data['alpha'] = mococomo_compute_responsibilities(
            self.data, self.f_list, self.logreg_list
        )
        self.elbo_history = []

    @property
    def responsibilities(self):
        """
        posterior assignment probability N x K matrix
        """
        return np.array(self.data['alpha'])

    @property
    def prior_log_odds(self):
        """
        return the (expected) prior log odds for each observation
        this is the (expectation of) the prediction from the logistic regression model
        """
        return np.array(self.logreg.predict())

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
        
    @property(self)
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
            self.data, self.f_list, self.logreg_list
        )

    def elbo(self, record: Boolean = False):
        new_elbo = mococomo_elbo(
            self.data, self.f_list, self.logreg_list, self.responsibilities
        )
        if record:
            self.elbo_history.append(new_elbo['total_elbo'])
        return new_elbo
    
    def update_responsibilities(self):
        """
        update the responsibilities and pass values to logreg
        """
        self.data['alpha'] = mococomo_compute_responsibilities(
            self.data, self.f_list, self.logreg_list
        )

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


    def update_f1(self):
        """
        Update the alternate component
        """
        self.f1.update(self.data)

    def iter(self):
        """
        iteration updates responsibilities, logistic regression, f0, and f1
        """
        self.update_responsibilities()
        self.logreg.update()
        self.f0.update(self.data)
        self.f1.update(self.data)


    def fit(self, niter=100, tol=1e-3):
        for i in range(niter):
            self.iter()
            self.elbo_history.append(self.elbo()['total_elbo'])
            if self.converged(tol):
                break

    def converged(self, tol=1e-3):
        return ((len(self.elbo_history) > 2)
            and ((self.elbo_history[-1] - self.elbo_history[-2]) < tol))

