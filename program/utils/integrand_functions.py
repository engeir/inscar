"""Script containing the integrands used in Gordeyev integrals.
"""

import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

from inputs import config as cf
from utils import v_int_parallel as para_int
from utils import vdfs


def ziebell_z_func(kappa, m, xi):
    F = sps.hyp2f1(1, 2 * kappa - 2 * m, kappa + m +
                   1, .5 * (1 + 1j * xi / kappa**.5))
    num = 1j * sps.gamma(kappa) * sps.gamma(kappa + m + 1 / 2)
    den = kappa**.5 * sps.gamma(kappa - .5) * sps.gamma(kappa + m + 1)
    Z = num / den * F
    return Z


# def two_p_isotropic_kappa(params):
#     w_bk = (2 * (params['kappa'] - 3 / 2) / params['kappa']
#             * params['T'] * const.k / params['m'])**.5
#     # (\zeta_\beta^0)
#     zbn = cf.w / (cf.K_RADAR * np.cos(cf.I_P['THETA']) * w_bk)
#     D = 2 * params['w_c']**2 / cf.w**2 * zbn**2 * \
#         ((params['kappa'] - .5) / params['kappa'] +
#          zbn * ziebell_z_func(params['kappa'], 1, zbn))
#     l_D2 = const.epsilon_0 * params['T'] / (cf.I_P['NE'] * const.elementary_charge**2) * (
#         params['kappa'] - 3 / 2) / (params['kappa'] - 1 / 2)
#     A = 1 / (cf.K_RADAR**2 * l_D2)
#     G = 1j * (1 - A * D) / cf.w
#     F = 1 - (1j * cf.w + params['nu']) * G
#     return F


def z_func(y, w_c, m, T, kappa):
    theta_2 = 2 * ((kappa - 3 / 2) / kappa) * T * const.k / m
    Z = (2 * kappa)**(1 / 2) * \
        (cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * theta_2 / w_c**2 *
         (1 - np.cos(w_c * y)) + 1 / 2 * cf.K_RADAR**2 * np.cos(cf.I_P['THETA'])**2 * theta_2 * y**2)**(1 / 2)
    return Z


def kappa_gordeyev(y, params):
    """Gordeyev integral for a kappa velocity distribution as defined by Mace (2003).

    The integral is valid for a plasma that is uniform, collisionless and permeated by a homogeneous,
    constant magnetic field. In the unperturbed state its intrinsic electric field vanishes.

    Arguments:
        y {np.ndarray} -- 1D array of the integration variable
        params {dict} -- a dictionary listing all plasma parameters needed to evaluate the integrand

    Returns:
        np.ndarray -- 1D array with the values of the integrand at the positions of the integration variable
    """
    z_value = z_func(y, params['w_c'], params['m'],
                     params['T'], params['kappa'])
    Kn = sps.kv(params['kappa'] + 1 / 2, z_value)
    Kn[Kn == np.inf] = 1
    G = z_value**(params['kappa'] + .5) * Kn * np.exp(- y * params['nu'])
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


def v_int(y, params):
    cf.SCALING = None
    v = np.linspace(0, cf.V_MAX**(1 / cf.ORDER), int(cf.V_N_POINTS))**cf.ORDER
    if params['vdf'] == 'maxwell':
        f = vdfs.f_0_maxwell
    elif params['vdf'] == 'kappa':
        f = vdfs.f_0_kappa
    elif params['vdf'] == 'kappa_vol2':
        f = vdfs.f_0_kappa_two
    elif params['vdf'] == 'gauss_shell':
        f = vdfs.f_0_gauss_shell
    elif params['vdf'] == 'real_data':
        f = vdfs.f_0_real_data

    res = para_int.integrand(y, params, v, f(v, params))
    return res


def v_int_integrand(y, params, v, f):
    sin = np.sin(p(y, params) * v)
    val = v * sin * f
    res = si.simps(val, v)
    return res


def p(y, params):
    k_perp = cf.K_RADAR * np.sin(cf.I_P['THETA'])
    k_par = cf.K_RADAR * np.cos(cf.I_P['THETA'])
    return (2 * k_perp**2 / params['w_c']**2 * (1 - np.cos(y * params['w_c'])) + k_par**2 * y**2)**.5


def p_d(y, params):
    # At y=0 we get 0/0, but in the limit as y tends to zero,
    # we get p_d = |k| * |w_c| / np.sqrt(w_c**2) (from above, opposite sign from below)
    cos_t = np.cos(cf.I_P['THETA'])
    sin_t = np.sin(cf.I_P['THETA'])
    w_c = params['w_c']
    num = abs(cf.K_RADAR) * abs(w_c) * (cos_t**2 *
                                        w_c * y + sin_t**2 * np.sin(w_c * y))
    den = w_c * (cos_t**2 * w_c**2 * y**2 - 2 * sin_t **
                 2 * np.cos(w_c * y) + 2 * sin_t**2)**.5
    # np.sign(y[-1]) takes care of weather the limit should be considered taken from above or below,
    # where the last element of the np.ndarray is chosen since it is assumed y runs from 0 to some finite real number.
    first = np.sign(y[-1]) * abs(cf.K_RADAR) * abs(w_c) / abs(w_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den
    out[np.where(den == 0.)[0]] = first

    return out


def long_calc(y, params):
    """Based on eq. (12) of Mace (2003).

    Arguments:
        y {np.ndarray} -- integration variable, 1D array of floats
        params {dict} -- dict of all plasma parameters

    Returns:
        np.ndarray -- the value of the integrand going into the integral in eq. (12)
    """
    return p_d(y, params) * v_int(y, params)
