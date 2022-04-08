"""Velocity distribution function to be used by the integrand class `a_vdf`.

One of the integrands available for use in the Gordeyev integral.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

from isr_spectrum import config


class Vdf(ABC):
    """Base class for a VDF object.

    Arguments:
        ABC {class} -- abstract base class that all VDF objects inherit from
    """

    @abstractmethod
    def normalize(self):
        """Calculate the normalization for the VDF."""

    @abstractmethod
    def f_0(self):
        """Return the values along the velocity axis of a VDF."""


class VdfMaxwell(Vdf):
    """Create an object that make Maxwellian distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make VDF objects
    """

    def __init__(self, params: config.Parameters, particle: config.Particle):
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self):
        self.A = (
            2 * np.pi * self.particle.temperature * const.k / self.particle.mass
        ) ** (-3 / 2)

    def f_0(self):
        return self.A * np.exp(
            -self.particle.velocity_axis**2
            / (2 * self.particle.temperature * const.k / self.particle.mass)
        )


class VdfKappa(Vdf):
    """Create an object that make kappa distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make VDF objects
    """

    def __init__(self, params: config.Parameters, particle: config.Particle):
        """Initialize VDF parameters.

        Arguments:
            v {np.ndarray} -- 1D array with the sampled velocities
            params {dict} -- a dictionary with all needed plasma parameters
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self):
        self.theta_2 = (
            2
            * ((self.particle.kappa - 3 / 2) / self.particle.kappa)
            * self.particle.temperature
            * const.k
            / self.particle.mass
        )
        self.A = (
            (np.pi * self.particle.kappa * self.theta_2) ** (-3 / 2)
            * sps.gamma(self.particle.kappa + 1)
            / sps.gamma(self.particle.kappa - 1 / 2)
        )

    def f_0(self):
        """Return the values along velocity `v` of a kappa VDF.

        Kappa VDF used in Gordeyev paper by Mace (2003).

        Returns:
            np.ndarray -- 1D array with the VDF values at the sampled points
        """
        return self.A * (
            1 + self.particle.velocity_axis**2 / (self.particle.kappa * self.theta_2)
        ) ** (-self.particle.kappa - 1)


class VdfKappa2(Vdf):
    """Create an object that make kappa vol. 2 distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make VDF objects
    """

    def __init__(self, v, params):
        """Initialize VDF parameters.

        Arguments:
            v {np.ndarray} -- 1D array with the sampled velocities
            params {dict} -- a dictionary with all needed plasma parameters
        """
        self.v = v
        self.params = params
        self.normalize()

    def normalize(self):
        self.v_th = np.sqrt(self.params["T"] * const.k / self.params["m"])
        self.A = (
            (np.pi * self.params["kappa"] * self.v_th**2) ** (-3 / 2)
            * sps.gamma(self.params["kappa"])
            / sps.gamma(self.params["kappa"] - 3 / 2)
        )

    def f_0(self):
        """Return the values along velocity `v` of a kappa VDF.

        Kappa VDF used in dispersion relation paper by
        Ziebell, Gaelzer and Simoes (2017). Defined by
        Leubner (2002) (sec 3.2).

        Returns:
            np.ndarray -- 1D array with the VDF values at the sampled points
        """
        return self.A * (1 + self.v**2 / (self.params["kappa"] * self.v_th**2)) ** (
            -self.params["kappa"]
        )


class VdfGaussShell(Vdf):
    """Create an object that make Gauss shell distribution functions.

    Arguments:
        VDF {ABC} -- abstract base class to make VDF objects
    """

    def __init__(self, v, params):
        self.v = v
        self.params = params
        self.vth = np.sqrt(self.params["T"] * const.k / self.params["m"])
        self.r = (self.params["T_ES"] * const.k / self.params["m"]) ** 0.5
        self.steep = 5
        self.f_M = VdfMaxwell(self.v, self.params)
        self.normalize()

    def normalize(self):
        func = np.exp(
            -self.steep
            * (abs(self.v) - self.r) ** 2
            / (2 * self.params["T"] * const.k / self.params["m"])
        )
        f = func * self.v**2 * 4 * np.pi
        self.A = 1 / si.simps(f, self.v)
        ev = 0.5 * const.m_e * self.r**2 / const.eV
        print(f"Gauss shell at E = {round(ev, 2)} eV")

    def f_0(self):
        func = (
            self.A
            * np.exp(
                -self.steep
                * (abs(self.v) - self.r) ** 2
                / (2 * self.params["T"] * const.k / self.params["m"])
            )
            + 1e4 * self.f_M.f_0()
        )

        return func / (1e4 + 1)