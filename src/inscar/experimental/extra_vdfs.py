"""Extra, experimental velocity distribution functions."""

import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

import inscar as isr


class VdfKappa2(isr.Vdf):
    """Create an object that make kappa vol. 2 distribution functions.

    Kappa VDF used in dispersion relation paper by Ziebell, Gaelzer and Simoes
    (2017). Defined by Leubner (2002) (sec 3.2).
    """

    def __init__(self, params: isr.Parameters, particle: isr.Particle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : Parameters
            `Parameters` object with the parameters of the simulation.
        particle : Particle
            `Particle` object with the parameters of the particle.
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self) -> None:
        self.v_th = np.sqrt(self.particle.temperature * const.k / self.particle.mass)
        self.A = (
            (np.pi * self.particle.kappa * self.v_th**2) ** (-3 / 2)
            * sps.gamma(self.particle.kappa)
            / sps.gamma(self.particle.kappa - 3 / 2)
        )

    def f_0(self) -> np.ndarray:
        return self.A * (
            1
            + self.particle.velocity_axis**2 / (self.particle.kappa * self.v_th**2)
        ) ** (-self.particle.kappa)


class VdfGaussShell(isr.Vdf):
    """Create an object that make Gauss shell distribution functions.

    This implementation need more testing and refining.
    """

    def __init__(self, params: isr.Parameters, particle: isr.Particle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : Parameters
            `Parameters` object with the parameters of the simulation.
        particle : Particle
            `Particle` object with the parameters of the particle.
        """
        self.params = params
        self.particle = particle
        temp_superthermal = getattr(self.particle, "temperature_superthermal", 90000)
        self.vth = np.sqrt(self.particle.temperature * const.k / self.particle.mass)
        self.r = (temp_superthermal * const.k / self.particle.mass) ** 0.5
        self.steep = 5
        self.f_M = isr.VdfMaxwell(self.params, self.particle)
        self.normalize()

    def normalize(self) -> None:
        func = np.exp(
            -self.steep
            * (abs(self.particle.velocity_axis) - self.r) ** 2
            / (2 * self.particle.temperature * const.k / self.particle.mass)
        )
        f = func * self.particle.velocity_axis**2 * 4 * np.pi
        self.A = 1 / si.simps(f, self.particle.velocity_axis)
        ev = 0.5 * const.m_e * self.r**2 / const.eV
        print(f"Gauss shell at E = {round(ev, 2)} eV")

    def f_0(self) -> np.ndarray:
        func = (
            self.A
            * np.exp(
                -self.steep
                * (abs(self.particle.velocity_axis) - self.r) ** 2
                / (2 * self.particle.temperature * const.k / self.particle.mass)
            )
            + 1e4 * self.f_M.f_0()
        )

        return func / (1e4 + 1)
