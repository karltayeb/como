# como (Covariate moderated EBNM)

Python implimentation of various covariate moderated EBNM

### Problem set-up

$$
\hat\beta \sim N(\beta, s^2) \\
\beta \sim \pi_0(x) f_0 + \pi_1(x) f_1 \\
\log \frac{\pi_1(x)}{\pi_0(x)} = {\bf \beta}^T {\bf x}
$$


1. A logistic regression model $\log \frac{\pi_1({\bf x})}{\pi_0({\bf x})} = {\bf b}^T {\bf x}
1. A null component distribution $f_0$
1. A alternative component distribution $f_1$

### TwoComponentCoMo

`TwoComponentCoMo` is an abstract class that encapsulates the logic for fitting two component covariate moderated EBNM.
It's constructor requires a dictionary 
1. `data` a dictionary
1. `f0` and `f1` two `ComponentDistribution` object (described below)
1. `logreg` a `LogisticRegression` object (described below)


### ComponentDistribution

We've implmented a few options for $f_0$ and $f_1$ that can be flexibly recombined. `ComponentDistribution`

1. `convolved_log_pdf(self, beta, se)`: compute $\log p(\hat\beta | s^2) = \log \int N(\hat\beta | \beta, s^2) f(\beta) d\beta$
1. `update(self, data)`: update the parameters of `ComponentDistribution` using `data`. Importantly, in practice this function should weight our observations according to the assignment probabilities.


So far we've implimented the following component distributions:

1. `PointMassComponent`: a point mass, usual choice for $f_0$
1. `NormalFixedLocComponent`: a normal distribution with fixed location parameter (default `loc=0`)-- we estimate the scale parameter
1. `UnimodalNormalMixtureComponent`: a scale mixture of normals, we estimate the mixture distribution $\pi$

### LogisticRegression

`LogisticRegression.predict()`: return (expected) predicted log odds given current parameter estimates
`LogisticRegression.evidence()`: compute the log likelihood or lower bound for current parameter estimates
`LogisticRegression.update()`: update the logistic regression, accounting for the current assignment probabilities

So far we've implimented the following classes inheriting `LogisticRegression`:

1. `LogisticSuSiE`: log odds are modeled with a sparse regression $\beta^T x$ where $\beta$ has the sum of single effects prior.
1. `InterceptOnly`: ignore the covariates, just estimate a constant mixture proprotion for all observations

### Putting it all together
We can easily specify a new two component model, inheriting from the base `TwoComponentCoMo` class

```
class PointNormalSuSiE(TwoComponentCoMo):
    def __init__(self, data, scale=1.0):
        """
        Initialize Point Normal SuSiE
        (Covariatiate EBNM with "point-normal" effects,
        and SuSiE prior on the mixture proportion)

        Parameters:
            data: dictionary with keys
                'beta' and 'se' for observations and standard errors,
                'X' and 'Z' for annotations and (fixed) covariates resp.
            scale: (initial) scale parameter for the normal mixture component
        """
        f0 = PointMassComponent(0.0)
        f1 = NormalFixedLocComponent(0, scale)
        
        # TODO: make sure `y` is a key in data, otherwise make it
        logreg = LogisticSusie(data, L=10)
        super().__init__(data, f0, f1, logreg)
```
