"""Implementation of integrals using `numba` for parallelization."""

import math
from typing import Optional

import numba as nb
import numpy as np
import scipy.constants as const

import inscar as isr


@nb.njit(parallel=True, cache=True)
def trapz(y, x):
    """Pure python version of trapezoid rule.

    Parameters
    ----------
    y : np.ndarray
        Function to integrate over.
    x : np.ndarray
        The axis to integrate along.

    Returns
    -------
    float
        The value of the integral.
    """
    s = 0
    for i in nb.prange(1, len(x)):
        s += (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return s / 2


@nb.njit(cache=True)
def inner_int(w, x, function):
    """Calculate the Gordeyev integral of the F function.

    Parameters
    ----------
    w : np.ndarray
        Angular frequency array.
    x : np.ndarray
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
        array[idx] = trapz(np.exp(-1j * w[idx] * x) * function, x)
    return array


def integrate(
    params: isr.Parameters,
    particle: isr.Particle,
    integrand: np.ndarray,
    the_type: str,
    char_vel: Optional[float] = None,
) -> np.ndarray:
    """Calculate the Gordeyev integral for each frequency.

    This locates the Gordeyev axis of the particle object, and for each frequency
    calculates the Gordeyev integral corresponding to it, returning an array of the same
    length as the frequency array.

    Parameters
    ----------
    params : isr.Parameters
        Parameters object with simulation parameters.
    particle : isr.Particle
        Particles object with particle parameters.
    integrand : np.ndarray
        Function to integrate over.
    the_type : str
        Defines which type of `isr.Integrand` class that is used.
    char_vel : float, optional
        Characteristic velocity of the particle.

    Returns
    -------
    np.ndarray
        The integral evaluated along the whole Gordeyev frequency axis of the particle.
    """
    y = particle.gordeyev_axis
    temp = particle.temperature
    mass = particle.mass
    w = params.angular_frequency
    nu = particle.collision_frequency
    array = inner_int(w, y, integrand)
    if the_type == "kappa":
        a = array / (2 ** (particle.kappa - 1 / 2) * math.gamma(particle.kappa + 1 / 2))
    elif the_type == "a_vdf":
        # Characteristic velocity scaling
        a = 4 * np.pi * temp * const.k * array / mass * char_vel
    else:
        a = array
    return a if the_type == "a_vdf" else 1 - (1j * w + nu) * a


@nb.njit(cache=True)
def integrate_velocity(y, v, f, k_r, theta, w_c):
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

    Returns
    -------
    np.ndarray
        The integrated function.
    """
    array = np.zeros_like(y)
    for idx in nb.prange(len(y)):
        array[idx] = trapz(v * np.sin(p(y[idx], k_r, theta, w_c) * v) * f, v)
    return array


@nb.njit(cache=True)
def p(y, k_r, theta, w_c):
    """From Mace [2003].

    Parameters
    ----------
    y: np.ndarray
        Parameter from Gordeyev integral
    k_r: float
        The radar wave number.
    theta: float
        The radar aspect angle.
    w_c: float
        The gyro frequency.

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
