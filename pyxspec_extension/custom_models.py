'''
Defines custom models for PyXSPEC to use.
'''

import math
import numpy as np
import xspec

from scipy.special import erfc


def exponentially_modified_gaussian(
    energy: float | np.ndarray,
    lambda_: float,
    mu: float,
    sigma: float
) -> float | np.ndarray:
    '''Function of an exponentially modified Gaussian.'''
    return (lambda_/2) * math.exp((lambda_/2) * (2*mu + lambda_*(sigma**2) - 2*energy)) * erfc((mu + lambda_*(sigma**2) - energy) / (1.4142135623730951 * sigma))


def expmodgauss(engs, params, flux):
    '''This is the XSPEC function definition.'''
    lam, mu, sigma, _ = params  # Fourth parameter is the normalization
    for i in range(len(engs)-1):
        flux[i] = exponentially_modified_gaussian(engs[i], lam, mu, sigma)


def _add_exponentially_modified_gaussian():
    '''Add the exponentially modified Gaussian to PyXSPEC.'''
    expmodgaussian_info = (
        'lam ct/keV/s 1.5 0.1 0.1 5. 5. 0.01',
        'mu keV 3. 2.5 2.5 10. 10. 0.01',
        'sigma keV 1. 0.01 0.01 5. 5. 0.01',
    )
    xspec.AllModels.addPyMod(expmodgauss, expmodgaussian_info, 'add')


def add_all_custom_models():
    '''Add all custom models to PyXSPEC.'''
    _add_exponentially_modified_gaussian()
