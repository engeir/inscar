import math

import numba as nb
import numpy as np
import scipy.constants as const


@nb.njit(parallel=True)
def trapzl(y, x):
    "Pure python version of trapezoid rule."
    s = 0
    for i in nb.prange(1, len(x)):
        s += (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return s / 2


@nb.njit(parallel=True)
def inner_int(y: np.ndarray, function: np.ndarray):
    """Calculate the Gordeyev integral of the F function.

    Parameters
    ----------
    y : np.ndarray
        The axis of the F function.
    function : np.ndarray
        Function to integrate over.

    Returns
    -------
    np.ndarray
        The integrated function.
    """
    array = np.zeros_like(cf.w, dtype=np.complex128)
    for idx in nb.prange(len(cf.w)):
        array[idx] = trapzl(np.exp(-1j * cf.w[idx] * y) * function, y)
    return array


def integrate(m, T, nu, y, function, kappa=1.0):
    array = inner_int(y, function.integrand())
    if function.the_type == "kappa":  # $\label{lst:gordeyev_scale}$
        a = array / (2 ** (kappa - 1 / 2) * math.gamma(kappa + 1 / 2))
    elif function.the_type == "a_vdf":
        # Characteristic velocity scaling
        a = 4 * np.pi * T * const.k * array / m * function.char_vel
    else:
        a = array
    return a if function.the_type == "a_vdf" else 1 - (1j * cf.w + nu) * a


@nb.njit(parallel=True)
def integrate_velocity(
    y: np.ndarray, v: np.ndarray, f: np.ndarray, k_r: float, theta: float, w_c: float
):
    """Calculate the velocity integral.

    Parameters
    ----------
    y : np.ndarray
        The axis in frequency space.
    v : np.ndarray
        The axis in velocity space.
    f : np.ndarray
        The function to integrate.
    k_r : float
        The radar wavenumber.
    theta : float
        The radar aspect angle.
    w_c : float
        The gyro frequency.
    """
    array = np.zeros_like(y)
    for idx in nb.prange(len(y)):
        array[idx] = trapzl(v * np.sin(p(y[idx], k_r, theta, w_c) * v) * f, v)
    return array


@nb.njit
def p(y, k_r, theta, w_c):
    """From Mace [2003].

    Parameters
    ----------
    y: np.ndarray
        Parameter from Gordeyev integral
    params: dict
        Plasma parameters

    Returns
    -------
    np.ndarray
        Value of the `p` function
    """
    k_perp = k_r * np.sin(theta)
    k_par = k_r * np.cos(theta)
    return (
        2 * k_perp ** 2 / w_c ** 2 * (1 - np.cos(y * w_c)) + k_par ** 2 * y ** 2
    ) ** 0.5
