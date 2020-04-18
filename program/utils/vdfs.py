"""Velocity distribution function used in the version long_calc, one of the integrands available for use in the Gordeyev integral.

Any new VDF must be added as an option in the long_calc function in integrand_functions.py.
"""

import numpy as np
import scipy.constants as const
import scipy.special as sps
import scipy.integrate as si
# import mpmath

from inputs import config as cf


def f_0_maxwell(v, params, module=np):
    A = (2 * module.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * module.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def f_0_kappa(v, params, module=np):
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
    A = (module.pi * params['kappa'] * theta_2)**(- 3 / 2) * \
        sps.gamma(params['kappa'] + 1) / sps.gamma(params['kappa'] - 1 / 2)
    func = A * (1 + v**2 / (params['kappa'] *
                            theta_2))**(- params['kappa'] - 1)
    return func


def f_0_kappa_two(v, params, module=np):
    """Return the values along velocity v of a kappa VDF.

    Kappa VDF used in dispersion relation paper by Ziebell, Gaelzer and Simoes (2017).
    Defined by Leubner (2002) (sec 3.2).

    Arguments:
        v {np.ndarray} -- 1D array with the sampled velocities
        params {dict} -- a dictionary with all needed plasma parameters

    Returns:
        np.ndarray -- 1D array with the VDF values at the sampled points
    """
    v_th = module.sqrt(params['T'] * const.k / params['m'])
    A = (module.pi * params['kappa'] * v_th**2)**(- 3 / 2) * \
        sps.gamma(params['kappa']) / sps.gamma(params['kappa'] - 3 / 2)
    func = A * (1 + v**2 / (params['kappa'] * v_th**2))**(- params['kappa'])
    return func


def make_gauss_shell_scaling(v, params, r, module=np):
    A = (2 * module.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * module.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m']))
    f = func * v**2 * 4 * module.pi
    res = si.simps(f, v)
    return res


def f_0_gauss_shell(v, params, module=np):
    vth = module.sqrt(params['T'] * const.k / params['m'])
    r = (cf.I_P['T_ES'] * const.k / params['m'])**.5
    print(f'Gauss shell at v = {round(r / vth,3)} v_th')
    # The radius of the shell will modify the area under
    # the curve to some extent and need proper scaling.
    if cf.SCALING is None:
        cf.SCALING = make_gauss_shell_scaling(v, params, r, module=module)
    A = (2 * module.pi * params['T'] * const.k / params['m'])**(- 3 / 2) / cf.SCALING
    func = A * module.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m'])) + \
        10 * f_0_maxwell(v, params, module=module)
    return func / 11
