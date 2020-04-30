"""Velocity distribution function used in the version long_calc, one of the integrands available for use in the Gordeyev integral.

Any new VDF must be added as an option in the long_calc function in integrand_functions.py.
"""

import numpy as np
import scipy.constants as const
import scipy.special as sps
import scipy.integrate as si

from inputs import config as cf
from data import read


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


def make_gauss_shell_scaling(v, params, r):
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * np.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m']))
    f = func * v**2 * 4 * np.pi
    res = si.simps(f, v)
    return res


def f_0_gauss_shell(v, params):
    vth = np.sqrt(params['T'] * const.k / params['m'])
    r = (cf.I_P['T_ES'] * const.k / params['m'])**.5
    print(f'Gauss shell at v = {round(r / vth,3)} v_th')
    # The radius of the shell will modify the area under
    # the curve to some extent and need proper scaling.
    if cf.SCALING is None:
        cf.SCALING = make_gauss_shell_scaling(v, params, r)
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2) / cf.SCALING
    func = A * np.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m'])) + \
        10 * f_0_maxwell(v, params)
    return func / 11


def scale_real_data(v, params):
    func = read.interpolate_data(v, params)
    f = func * v**2 * 4 * np.pi
    res = si.simps(f, v)
    return res


def f_0_real_data(v, params):
    if cf.SCALING is None:
        cf.SCALING = scale_real_data(v, params)
    func = read.interpolate_data(v, params) / cf.SCALING
    return func
