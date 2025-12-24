import numpy as np
from scipy.stats import norm

def area_gaussian_outside(mu: float,
                          var: float,
                          E_minus: float,
                          E_plus: float,
                         ) -> float:
    """
    Return P(X < E_minus) + P(X > E_plus)
    for X ~ N(mu, var).
    """
    sigma = np.sqrt(var)
    cdf_minus = norm.cdf(E_minus, loc=mu, scale=sigma)
    cdf_plus  = norm.cdf(E_plus,  loc=mu, scale=sigma)
    return cdf_minus + (1.0 - cdf_plus)