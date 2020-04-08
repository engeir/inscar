import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps
import mpmath as mpm
from tqdm import tqdm

import config as cf
import v_int_parallel as para_int
import int_cy


def ziebell_z_func(kappa, m, xi):
    F = sps.hyp2f1(1, 2 * kappa - 2 * m, kappa + m +
                   1, .5 * (1 + 1j * xi / kappa**.5))
    num = 1j * sps.gamma(kappa) * sps.gamma(kappa + m + 1 / 2)
    den = kappa**.5 * sps.gamma(kappa - .5) * sps.gamma(kappa + m + 1)
    Z = num / den * F
    return Z


def two_p_isotropic_kappa(params):
    w_bk = (2 * (params['kappa'] - 3 / 2) / params['kappa']
            * params['T'] * const.k / params['m'])**.5
    # (\zeta_\beta^0)
    zbn = cf.w / (cf.K_RADAR * np.cos(cf.I_P['THETA']) * w_bk)
    D = 2 * params['w_c']**2 / cf.w**2 * zbn**2 * \
        ((params['kappa'] - .5) / params['kappa'] +
         zbn * ziebell_z_func(params['kappa'], 1, zbn))
    l_D2 = const.epsilon_0 * params['T'] / (cf.I_P['NE'] * const.elementary_charge**2) * (
        params['kappa'] - 3 / 2) / (params['kappa'] - 1 / 2)
    A = 1 / (cf.K_RADAR**2 * l_D2)
    G = 1j * (1 - A * D) / cf.w
    F = 1 - (1j * cf.w + params['nu']) * G
    return F


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
    # * np.exp(- y * (- params['kappa'] - 1 / 2))
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


def f_0_maxwell(v, params):
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * np.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def f_0_kappa(v, params):
    """Return the values along velocity v of a kappa VDF.

    Kappa VDF used in Gordeyev paper by Mace (2003).

    Arguments:
        v {np.ndarray} -- 1D array with the sampled velocities
        params {dict} -- a dictionary with all needed plasma parameters

    Returns:
        np.ndarray -- 1D array with the VDF values at the sampled points
    """
    theta_2 = 2 * ((params['kappa'] - 3 / 2) / params['kappa']
                   ) * params['T'] * const.k / params['m']
    A = (np.pi * params['kappa'] * theta_2)**(- 3 / 2) * \
        sps.gamma(params['kappa'] + 1) / sps.gamma(params['kappa'] - 1 / 2)
    func = A * (1 + v**2 / (params['kappa'] *
                            theta_2))**(- params['kappa'] - 1)
    return func


def f_0_kappa_two(v, params):
    """Return the values along velocity v of a kappa VDF.

    Kappa VDF used in dispersion relation paper by Ziebell, Gaelzer and Simoes (2017).
    Defined by Leubner (2002) (sec 3.2).

    Arguments:
        v {np.ndarray} -- 1D array with the sampled velocities
        params {dict} -- a dictionary with all needed plasma parameters

    Returns:
        np.ndarray -- 1D array with the VDF values at the sampled points
    """
    v_th = np.sqrt(params['T'] * const.k / params['m'])
    A = (np.pi * params['kappa'] * v_th**2)**(- 3 / 2) * \
        sps.gamma(params['kappa']) / sps.gamma(params['kappa'] - 3 / 2)
    func = A * (1 + v**2 / (params['kappa'] * v_th**2))**(- params['kappa'])
    return func


def f_0_gauss_shell(v, params):
    vth = np.sqrt(params['T'] * const.k / params['m'])
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2) / 2
    func = A * np.exp(- (np.sqrt(v**2) - 5 * vth)**2 / (2 * params['T'] * const.k / params['m'])) + \
        10 * f_0_maxwell(v, params)
    return func / 11


def mpm_f_0_maxwell(v, params):
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * mpm.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def vv_int(params, j, v):
    return mpm_f_0_maxwell(v, params) * v * mpm.sin(mpm_p(j, params) * v)


def v_int(y, params):
    V_MAX = 1e7
    v = np.linspace(0, V_MAX**(1 / cf.ORDER), int(cf.N_POINTS))**cf.ORDER
    if params['vdf'] == 'maxwell':
        f = f_0_maxwell(v, params)
    elif params['vdf'] == 'kappa':
        f = f_0_kappa(v, params)
    elif params['vdf'] == 'kappa_vol2':
        f = f_0_kappa_two(v, params)
    elif params['vdf'] == 'gauss_shell':
        f = f_0_gauss_shell(v, params)

    res = para_int.integrand(y, params, v, f)
    return res


def v_int_integrand(j, params, v, f):
    sin = np.sin(p(j, params) * v)
    val = v * sin * f
    res = si.simps(val, v)
    return res


def p(y, params):
    k_perp = cf.K_RADAR * np.sin(cf.I_P['THETA'])
    k_par = cf.K_RADAR * np.cos(cf.I_P['THETA'])
    return (2 * k_perp**2 / params['w_c']**2 * (1 - np.cos(y * params['w_c'])) + k_par**2 * y**2)**.5


def mpm_p(y, params):
    k_perp = cf.K_RADAR * mpm.sin(cf.I_P['THETA'])
    k_par = cf.K_RADAR * mpm.cos(cf.I_P['THETA'])
    return (2 * k_perp**2 / params['w_c']**2 * (1 - mpm.cos(y * params['w_c'])) + k_par**2 * y**2)**.5


def p_d(y, params):
    # At y=0 we get 0/0, but in the limit as y tends to zero, we get frac = |k| * |w_c| / np.sqrt(w_c**2) (from above, opposite sign from below)
    cos_t = np.cos(cf.I_P['THETA'])
    sin_t = np.sin(cf.I_P['THETA'])
    w_c = params['w_c']
    num = abs(cf.K_RADAR) * abs(w_c) * (cos_t**2 *
                                        w_c * y + sin_t**2 * np.sin(w_c * y))
    den = w_c * (cos_t**2 * w_c**2 * y**2 - 2 * sin_t **
                 2 * np.cos(w_c * y) + 2 * sin_t**2)**.5
    first = np.sign(y[-1]) * abs(cf.K_RADAR) * abs(w_c) / np.sqrt(w_c**2)
    den[np.where(den == 0.)[0]] = first
    out = num / den

    return out


# params = {'nu': Lambda_s * w_c, 'm': m, 'T': T, 'w_c': w_c, 'kappa': kappa}
def long_calc(y, params):
    """Based on eq. (12) of Mace (2003).

    Arguments:
        y {np.ndarray} -- integration variable, 1D array of floats
        params {dict} -- dict of all plasma parameters

    Returns:
        np.ndarray -- the value of the integrand going into the integral in eq. (12)
    """
    return p_d(y, params) * v_int(y, params)
    # return p_d(y, params) * int_cy.v_int(y, params)
