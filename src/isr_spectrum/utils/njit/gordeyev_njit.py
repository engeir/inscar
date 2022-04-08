import math

import numba as nb
import numpy as np
import scipy.constants as const

from isr_spectrum.utils import config


@nb.njit(parallel=True)
def trapzl(y, x):
    "Pure python version of trapezoid rule."
    s = 0
    for i in nb.prange(1, len(x)):
        s += (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return s / 2


@nb.njit()
def inner_int(w: np.ndarray, y: np.ndarray, function: np.ndarray):
    """Calculate the Gordeyev integral of the F function.

    Parameters
    ----------
    w : ndarray
        Angular frequency array.
    y : np.ndarray
        The axis of the F function.
    function : np.ndarray
        Function to integrate over.

    Returns
    -------
    np.ndarray
        The integrated function.
    """
    array = np.zeros_like(w, dtype=np.complex128)
    for idx in nb.prange(len(w)):
        array[idx] = trapzl(np.exp(-1j * w[idx] * y) * function, y)
    return array


def integrate(
    params: config.Parameters,
    particle: config.Particle,
    integrand: np.ndarray,
    the_type: str,
    char_vel: float,
):
    y = particle.gordeyev_axis
    temp = particle.temperature
    mass = particle.mass
    w = params.angular_frequency
    nu = particle.collision_frequency
    array = inner_int(w, y, integrand)
    if the_type == "kappa":  # $\label{lst:gordeyev_scale}$
        a = array / (2 ** (particle.kappa - 1 / 2) * math.gamma(particle.kappa + 1 / 2))
    elif the_type == "a_vdf":
        # Characteristic velocity scaling
        a = 4 * np.pi * temp * const.k * array / mass * char_vel
    else:
        a = array
    return a if the_type == "a_vdf" else 1 - (1j * w + nu) * a


@nb.njit()
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
        2 * k_perp**2 / w_c**2 * (1 - np.cos(y * w_c)) + k_par**2 * y**2
    ) ** 0.5
