import scipy.constants as const
import scipy.special as sps
import numpy as np

import config as cf


def z_func(y, w_c, m, T):
    theta_2 = 2 * ((cf.KAPPA - 3 / 2) / cf.KAPPA) * T * const.k / m
    Z = (2 * cf.KAPPA)**(1 / 2) * (cf.K_RADAR**2 * np.sin(cf.THETA)**2 * theta_2 / w_c**2 *
                                   (1 - np.cos(w_c * y)) + 1 / 2 * cf.K_RADAR**2 * np.cos(cf.THETA)**2 * theta_2 * y**2)**(1 / 2)
    return Z


def kappa_gordeyev(y, params):
    z_value = z_func(y, params['w_c'], params['m'], params['T'])
    Kn = sps.kn(cf.KAPPA + 1 / 2, z_value)
    Kn[Kn == np.inf] = 1
    G = z_value**(cf.KAPPA + .5) * Kn * np.exp(- y * cf.NU)
    return G


def maxwell_gordeyev(y, params):
    G = np.exp(- y * params['nu'] -
               cf.K_RADAR**2 * np.sin(cf.THETA)**2 * params['T'] * const.k / (params['m'] * params['w_c']**2) *
               (1 - np.cos(params['w_c'] * y)) - .5 * (cf.K_RADAR * np.cos(cf.THETA) * y)**2 * params['T'] * const.k / params['m'])
    return G


def F_s_integrand(y, params):  # X_s, Lambda_s):
    """Calculate the integral in the expression for F_i in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    W = np.exp(- params['nu'] * y -
               (params['T'] * const.k * cf.K_RADAR**2 / (params['m'] * params['w_c']**2)) *
               (np.sin(cf.THETA)**2 * (1 - np.cos(params['w_c'] * y)) + 1 / 2 * params['w_c']**2 * np.cos(cf.THETA)**2 * y**2))
    return W
