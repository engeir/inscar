"""Script containing the integrands used in Gordeyev integrals.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.constants as const
import scipy.special as sps

from inputs import config as cf
from utils import v_int_parallel as para_int
from utils import vdfs


class INTEGRAND(ABC):
    @abstractmethod
    def initialize(self, y, params):
        pass

    @abstractmethod
    def integrand(self):
        pass


class INT_KAPPA(INTEGRAND):
    def __init__(self):
        self.y = np.array([])
        self.params = {}
        self.type = 'kappa'
        self.Z = float
        self.Kn = float

    def initialize(self, y, params):
        self.y = y
        self.params = params
        self.z_func()

    def z_func(self):
        theta_2 = 2 * ((self.params['kappa'] - 3 / 2) / self.params['kappa']) * self.params['T'] * const.k / self.params['m']
        self.Z = (2 * self.params['kappa'])**(1 / 2) * \
            (cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * theta_2 / self.params['w_c']**2 *
             (1 - np.cos(self.params['w_c'] * self.y)) +
             1 / 2 * cf.K_RADAR**2 * np.cos(cf.I_P['THETA'])**2 * theta_2 * self.y**2)**(1 / 2)
        self.Kn = sps.kv(self.params['kappa'] + 1 / 2, self.Z)
        self.Kn[self.Kn == np.inf] = 1

    def integrand(self):
        G = self.Z**(self.params['kappa'] + .5) * self.Kn * np.exp(- self.y * self.params['nu'])

        return G

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


class INT_MAXWELL(INTEGRAND):
    def __init__(self):
        self.y = np.array([])
        self.params = {}
        self.type = 'maxwell'

    def initialize(self, y, params):
        self.y = y
        self.params = params

    def integrand(self):
        G = np.exp(- self.y * self.params['nu'] -
                   cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * self.params['T'] * const.k /
                   (self.params['m'] * self.params['w_c']**2) * (1 - np.cos(self.params['w_c'] * self.y)) -
                   .5 * (cf.K_RADAR * np.cos(cf.I_P['THETA']) * self.y)**2 * self.params['T'] * const.k / self.params['m'])

        return G

def maxwell_gordeyev(y, params):
    """Calculate the integral in the expression for F_i in Hagfors, i.e. according to a Maxwellian distribution.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    G = np.exp(- y * params['nu'] -
               cf.K_RADAR**2 * np.sin(cf.I_P['THETA'])**2 * params['T'] * const.k /
               (params['m'] * params['w_c']**2) * (1 - np.cos(params['w_c'] * y)) -
               .5 * (cf.K_RADAR * np.cos(cf.I_P['THETA']) * y)**2 * params['T'] * const.k / params['m'])
    return G


# def F_s_integrand(y, params):
#     """Calculate the integral in the expression for F_i in Hagfors.

#     Arguments:
#         y {float} -- integration variable

#     Returns:
#         float -- the value of the integral
#     """
#     W = np.exp(- params['nu'] * y -
#                (params['T'] * const.k * cf.K_RADAR**2 / (params['m'] * params['w_c']**2)) *
#                (np.sin(cf.I_P['THETA'])**2 * (1 - np.cos(params['w_c'] * y)) +
#                 1 / 2 * params['w_c']**2 * np.cos(cf.I_P['THETA'])**2 * y**2))
#     return W


class INT_LONG(INTEGRAND):
    def __init__(self):
        self.y = np.array([])
        self.params = {}
        self.type = 'long_calc'

    def initialize(self, y, params):
        self.y = y
        self.params = params

    def v_int(self):
        v = np.linspace(0, cf.V_MAX**(1 / cf.ORDER), int(cf.V_N_POINTS))**cf.ORDER
        if self.params['vdf'] == 'maxwell':
            # f = vdfs.f_0_maxwell
            f = vdfs.F_MAXWELL(v, self.params)
        elif self.params['vdf'] == 'kappa':
            # f = vdfs.f_0_kappa
            f = vdfs.F_KAPPA(v, self.params)
        elif self.params['vdf'] == 'kappa_vol2':
            # f = vdfs.f_0_kappa_two
            f = vdfs.F_KAPPA_2(v, self.params)
        elif self.params['vdf'] == 'gauss_shell':
            # f = vdfs.f_0_gauss_shell
            f = vdfs.F_GAUSS_SHELL(v, self.params)
        elif self.params['vdf'] == 'real_data':
            # f = vdfs.f_0_real_data
            f = vdfs.F_REAL_DATA(v, self.params)

        # res = para_int.integrand(y, params, v, f(v, params))
        res = para_int.integrand(self.y, self.params, v, f.f_0())
        return res

    def p_d(self):
        # At y=0 we get 0/0, but in the limit as y tends to zero,
        # we get p_d = |k| * |w_c| / np.sqrt(w_c**2) (from above, opposite sign from below)
        cos_t = np.cos(cf.I_P['THETA'])
        sin_t = np.sin(cf.I_P['THETA'])
        w_c = self.params['w_c']
        num = abs(cf.K_RADAR) * abs(w_c) * (cos_t**2 *
                                            w_c * self.y + sin_t**2 * np.sin(w_c * self.y))
        den = w_c * (cos_t**2 * w_c**2 * self.y**2 - 2 * sin_t **
                     2 * np.cos(w_c * self.y) + 2 * sin_t**2)**.5
        # np.sign(y[-1]) takes care of weather the limit should be considered taken from above or below,
        # where the last element of the np.ndarray is chosen since it is assumed y runs from 0 to some finite real number.
        first = np.sign(self.y[-1]) * abs(cf.K_RADAR) * abs(w_c) / abs(w_c)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num / den
        out[np.where(den == 0.)[0]] = first

        return out

    def integrand(self):
        return self.p_d() * self.v_int()


# def v_int(y, params):
#     # cf.SCALING = None
#     v = np.linspace(0, cf.V_MAX**(1 / cf.ORDER), int(cf.V_N_POINTS))**cf.ORDER
#     if params['vdf'] == 'maxwell':
#         # f = vdfs.f_0_maxwell
#         f = vdfs.F_MAXWELL(v, params)
#     elif params['vdf'] == 'kappa':
#         # f = vdfs.f_0_kappa
#         f = vdfs.F_KAPPA(v, params)
#     elif params['vdf'] == 'kappa_vol2':
#         # f = vdfs.f_0_kappa_two
#         f = vdfs.F_KAPPA_2(v, params)
#     elif params['vdf'] == 'gauss_shell':
#         # f = vdfs.f_0_gauss_shell
#         f = vdfs.F_GAUSS_SHELL(v, params)
#     elif params['vdf'] == 'real_data':
#         # f = vdfs.f_0_real_data
#         f = vdfs.F_REAL_DATA(v, params)

#     # res = para_int.integrand(y, params, v, f(v, params))
#     res = para_int.integrand(y, params, v, f.f_0())
#     return res


# def v_int_integrand(y, params, v, f):
#     sin = np.sin(p(y, params) * v)
#     val = v * sin * f
#     res = si.simps(val, v)
#     return res


# def p(y, params):
#     k_perp = cf.K_RADAR * np.sin(cf.I_P['THETA'])
#     k_par = cf.K_RADAR * np.cos(cf.I_P['THETA'])
#     return (2 * k_perp**2 / params['w_c']**2 * (1 - np.cos(y * params['w_c'])) + k_par**2 * y**2)**.5


# def p_d(y, params):
#     # At y=0 we get 0/0, but in the limit as y tends to zero,
#     # we get p_d = |k| * |w_c| / np.sqrt(w_c**2) (from above, opposite sign from below)
#     cos_t = np.cos(cf.I_P['THETA'])
#     sin_t = np.sin(cf.I_P['THETA'])
#     w_c = params['w_c']
#     num = abs(cf.K_RADAR) * abs(w_c) * (cos_t**2 *
#                                         w_c * y + sin_t**2 * np.sin(w_c * y))
#     den = w_c * (cos_t**2 * w_c**2 * y**2 - 2 * sin_t **
#                  2 * np.cos(w_c * y) + 2 * sin_t**2)**.5
#     # np.sign(y[-1]) takes care of weather the limit should be considered taken from above or below,
#     # where the last element of the np.ndarray is chosen since it is assumed y runs from 0 to some finite real number.
#     first = np.sign(y[-1]) * abs(cf.K_RADAR) * abs(w_c) / abs(w_c)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         out = num / den
#     out[np.where(den == 0.)[0]] = first

#     return out


# def long_calc(y, params):
#     """Based on eq. (12) of Mace (2003).

#     Arguments:
#         y {np.ndarray} -- integration variable, 1D array of floats
#         params {dict} -- dict of all plasma parameters

#     Returns:
#         np.ndarray -- the value of the integrand going into the integral in eq. (12)
#     """
#     return p_d(y, params) * v_int(y, params)
