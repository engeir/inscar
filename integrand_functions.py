import scipy.constants as const
import scipy.special as sps
import numpy as np

import config as cf


def ziebell_z_func(kappa, m, xi):
    F = sps.hyp2f1(1, 2 * kappa - 2 * m, kappa + m +
                   1, .5 * (1 + 1j * xi / kappa**.5))
    num = 1j * sps.gamma(kappa) * sps.gamma(kappa + m + 1 / 2)
    den = kappa**.5 * sps.gamma(kappa - .5) * sps.gamma(kappa + m + 1)
    Z = num / den * F
    return Z


def two_p_isotropic_kappa(params):
    w_bk = ((cf.KAPPA - 3 / 2) / cf.KAPPA)**.5 * params['T'] * const.k / params['m']
    zbn = cf.w / (cf.K_RADAR * np.cos(cf.I_P['THETA']) * w_bk)  # (\zeta_\beta^0)
    D = 2 * params['w_c']**2 / cf.w**2 * zbn**2 * \
        ((cf.KAPPA - .5) / cf.KAPPA + zbn * ziebell_z_func(cf.KAPPA, 1, zbn))
    l_D2 = const.epsilon_0 * params['T'] / (cf.I_P['NE'] * const.elementary_charge**2) * (cf.KAPPA - 3 / 2) / (cf.KAPPA - 1 / 2)
    A = 1 / (cf.K_RADAR**2 * l_D2)
    G = 1j * (1 - A * D) / cf.w
    F = 1 - (1j * cf.w + params['nu']) * G
    return F


def z_func(y, w_c, m, T):
    theta_2 = 2 * ((cf.KAPPA - 3 / 2) / cf.KAPPA) * T * const.k / m
    Z = (2 * cf.KAPPA)**(1 / 2) * \
        (cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * theta_2 / w_c**2 *
         (1 - np.cos(w_c * y)) + 1 / 2 * cf.K_RADAR**2 * np.cos(cf.I_P['THETA'])**2 * theta_2 * y**2)**(1 / 2)
    return Z


def kappa_gordeyev(y, params):
    z_value = z_func(y, params['w_c'], params['m'], params['T'])
    Kn = sps.kv(cf.KAPPA + 1 / 2, z_value)
    Kn[Kn == np.inf] = 1
    G = z_value**(cf.KAPPA + .5) * Kn * np.exp(- y * cf.NU)
    return G


def maxwell_gordeyev(y, params):
    G = np.exp(- y * params['nu'] -
               cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * params['T'] * const.k /
               (params['m'] * params['w_c']**2) * (1 - np.cos(params['w_c'] * y)) -
               .5 * (cf.K_RADAR * np.cos(cf.I_P['THETA']) * y)**2 * params['T'] * const.k / params['m'])
    return G


def F_s_integrand(y, params):
    """Calculate the integral in the expression for F_i in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    W = np.exp(- params['nu'] * y -
               (params['T'] * const.k * cf.K_RADAR**2 / (params['m'] * params['w_c']**2)) *
               (np.sin(cf.I_P['THETA'])**2 * (1 - np.cos(params['w_c'] * y)) +
                1 / 2 * params['w_c']**2 * np.cos(cf.I_P['THETA'])**2 * y**2))
    return W
