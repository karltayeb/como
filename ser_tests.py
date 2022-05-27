import numpy as np
from como.utils import *
from como.logistic_ser import *
from como.logistic_susie import *

def report(params):
    intercept = params['delta'][0]
    map_idx = params['alpha'].argmax()
    map_beta = params['mu'][map_idx]
    print(f'b0: {intercept}, b1: {map_beta}, idx: {map_idx}')

def sim_ser(n, p, b0, b1):
    X = np.random.binomial(1, 0.02, size=n*p).reshape(n, -1)
    X[:, :9] = X[:, 9][:, None]
    y = np.random.binomial(1, sigmoid(b0 + b1 * X[:, 9]))
    data = {
        'y': y,
        'X': X,
        'Z': np.ones((n, 1)),
        'params': dict(b0=b0, b1=b1)
    }
    return data

def sim_susie(n, p, b0, b):
    #mix = np.exp(-np.abs(np.arange(p) - np.arange(p)[:, None]) / 3)
    #X = np.random.binomial(1, 0.2, size=n*p).reshape(n, -1) @ mix
    X = np.random.normal(size=n*p).reshape(n, -1) #@ mix
    X[:, :9] = X[:, 9][:, None]
    X[:, 10:19] = X[:, 19][:, None]
    X[:, 20:29] = X[:, 29][:, None]

    p = jnp.clip(sigmoid(b0 + X[:, [9, 19, 29]] @ b), 1e-8, 1-1e-8)

    y = np.random.binomial(1, p)
    data = {
        'y': y,
        'p' : p,
        'X': X,
        'Z': np.ones((n, 1)),
        'params': dict(b0=b0, b=b)
    }
    return data

def test_sim_ser():
    data = sim_ser(1000, 100, -3, 4)
    offset = np.zeros(1000)


    params, hypers = None, None
    for i in range(5):
        params, _ = fit_ser(data, params, hypers, offset, niter=50)
        print(elbo_ser(data, params, hypers, offset))

    report(params)
    cs = get_credible_set(fit_params)
    print(cs)

def test_sim_susie():
    L = 4
    data = sim_susie(1000, 100, -4, np.array([2., 4., 6.]))
    params, hypers = susie_fit(data, 10, 10)
    assert False
