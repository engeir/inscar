import numpy as np
import scipy.constants as const
import scipy.special as sps
import scipy.integrate as si


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
    vth = np.sqrt(params['T'] * const.k / params['m'])
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2)
    func = A * np.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m']))
    f = func * v**2 * 4 * np.pi
    res = si.simps(f, v)
    return res


def f_0_gauss_shell(v, params):
    vth = np.sqrt(params['T'] * const.k / params['m'])
    r = 5 * vth
    scaling = make_gauss_shell_scaling(v, params, r)  # 51.87353390551461
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 3 / 2) / scaling
    func = A * np.exp(- (abs(v) - r)**2 / (2 * params['T'] * const.k / params['m'])) + \
        10 * f_0_maxwell(v, params)
    return func / 11
