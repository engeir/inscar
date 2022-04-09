"""Velocity distribution function to be used by the integrand class `a_vdf`.

One of the integrands available for use in the Gordeyev integral.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.constants as const
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
    """Create an object that make kappa distribution functions."""

    def __init__(self, params: config.Parameters, particle: config.Particle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : Parameters
            Parameters object with the parameters of the simulation.
        particle : Particle
            Particle object with the parameters of the particle.
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

    def f_0(self) -> np.ndarray:
        """Return the values along velocity `v` of a kappa VDF.

        Kappa VDF used in Gordeyev paper by Mace (2003).

        Returns
        -------
        np.ndarray
            1D array with the VDF values at the sampled points
        """
        return self.A * (
            1 + self.particle.velocity_axis**2 / (self.particle.kappa * self.theta_2)
        ) ** (-self.particle.kappa - 1)
