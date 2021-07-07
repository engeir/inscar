import math

import numba as nb
import numpy as np
import scipy.constants as const

from isr_spectrum.inputs import config as cf


@nb.njit(parallel=True)
def trapzl(y, x):
    "Pure python version of trapezoid rule."
    s = 0
    for i in nb.prange(1, len(x)):
        s += (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return s / 2


@nb.njit(parallel=True)
def my_integration_function(w, y, f):
    val = np.exp(-1j * w * y) * f

    return trapzl(val, y)


@nb.njit(parallel=True)
def inner_int(y, function):
    # f = function  # function.integrand()
    array = np.zeros_like(cf.w, dtype=np.complex128)
    # a = np.zeros_like(cf.w, dtype=np.complex128)
    # F = np.zeros_like(cf.w, dtype=np.complex128)
    for idx in nb.prange(len(cf.w)):
        array[idx] = my_integration_function(cf.w[idx], y, function)
    return array


def integrate(m, T, nu, y, function, the_type: str, kappa=1.0):
    # f = function  # function.integrand()
    # array = np.zeros_like(cf.w, dtype=np.complex128)
    # a = np.zeros_like(cf.w, dtype=np.complex128)
    # F = np.zeros_like(cf.w, dtype=np.complex128)
    # for idx in nb.prange(len(cf.w)):
    #     array[idx] = my_integration_function(cf.w[idx], y, function)
    array = inner_int(y, function.integrand())
    if the_type == "kappa":  # $\label{lst:gordeyev_scale}$
        a = array / (2 ** (kappa - 1 / 2) * math.gamma(kappa + 1 / 2))
        # a = array / (2 ** (kappa - 1 / 2) * sps.gamma(kappa + 1 / 2))
    elif the_type == "a_vdf":
        # Characteristic velocity scaling
        a = 4 * np.pi * T * const.k * array / m * function.char_vel
        # a = 4 * np.pi * T * const.k * array / m * function.char_vel
    else:
        a = array
    if the_type == "a_vdf":
        F = a
    else:
        F = 1 - (1j * cf.w + nu) * a
    return F


@nb.njit
def v_int_integrand(y, v, f, k_r, theta, w_c):
    sin = np.sin(p(y, k_r, theta, w_c) * v)
    val = v * sin * f
    # res = si.simps(val, v)
    res = trapzl(val, v)
    return res


@nb.njit(parallel=True)
def integrate_velocity(y, v, f, k_r, theta, w_c):
    array = np.zeros_like(y)
    for idx in nb.prange(len(y)):
        array[idx] = v_int_integrand(y[idx], v, f, k_r, theta, w_c)
    return array


@nb.njit
def p(y, k_r, theta, w_c):
    """From Mace [2003].

    Args:
        y {np.ndarray} -- parameter from Gordeyev integral
        params {dict} -- plasma parameters

    Returns:
        np.ndarray -- value of the `p` function
    """
    k_perp = k_r * np.sin(theta)
    k_par = k_r * np.cos(theta)
    return (
        2 * k_perp ** 2 / w_c ** 2 * (1 - np.cos(y * w_c)) + k_par ** 2 * y ** 2
    ) ** 0.5
