"""Script containing the calculation of the power density spectrum
and other plasma parameters.
"""

from typing import Tuple

import numpy as np
import scipy.constants as const

from isr_spectrum.utils import config
from isr_spectrum.utils import integrand_functions as intf
from isr_spectrum.utils.njit import gordeyev_njit


class SpectrumCalculation:
    """Class containing the calculation of the power density spectrum."""

    def __init__(self):
        self.ion: config.Particle
        self.electron: config.Particle
        self.ion_integration_function: intf.Integrand
        self.electron_integration_function: intf.Integrand
        self._calulate_f = self._calulate_f_function
        self._susceptibility = self._susceptibility_function
        self.params: config.Parameters

    def set_params(self, params: config.Parameters) -> None:
        self.params = params

    def set_ion(self, ion: config.Particle) -> None:
        self.ion = ion

    def set_electron(self, electron: config.Particle) -> None:
        self.electron = electron

    def set_ion_integration_function(self, function: intf.Integrand) -> None:
        self.ion_integration_function = function

    def set_electron_integration_function(self, function: intf.Integrand) -> None:
        self.electron_integration_function = function

    def calculate_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "params"):
            raise ValueError("No parameters set. Use set_params().")
        if not hasattr(self, "ion"):
            raise ValueError("No ion particle set. Use set_ion().")
        if not hasattr(self, "electron"):
            raise ValueError("No electron particle set. Use set_electron().")
        if not hasattr(self, "ion_integration_function"):
            raise ValueError(
                "No ion integration function set. Use set_ion_integration_function()."
            )
        if not hasattr(self, "electron_integration_function"):
            raise ValueError(
                "No electron integration function set. Use set_electron_integration_function()."
            )

        fi = self._calulate_f(self.ion, self.ion_integration_function)
        fe = self._calulate_f(self.electron, self.electron_integration_function)

        xp_i = self._susceptibility(self.ion, self.ion_integration_function)
        xp_e = self._susceptibility(self.electron, self.electron_integration_function)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator1 = np.imag(-fe) * np.abs(1 + 2 * xp_i**2 * fi) ** 2
            numerator2 = 4 * xp_e**4 * np.imag(-fi) * np.abs(fe) ** 2
            numerator = numerator1 + numerator2
            denominator = np.abs(1 + 2 * xp_e**2 * fe + 2 * xp_i**2 * fi) ** 2
            spectrum = (
                self.electron.number_density
                / (np.pi * self.params.angular_frequency)
                * numerator
                / denominator
            )
        return self.params.linear_frequency, spectrum

    def set_calculate_f_function(self, f_func) -> None:
        self._calulate_f = f_func

    def _calulate_f_function(
        self, particle: config.Particle, int_func: intf.Integrand
    ) -> np.ndarray:
        int_func.initialize(self.params, particle)
        the_type = int_func.the_type
        integrand = int_func.integrand()
        characteristic_velocity = getattr(int_func, "char_vel", None)
        return gordeyev_njit.integrate(
            self.params,
            particle,
            integrand,
            the_type,
            characteristic_velocity,
        )

    def set_susceptibility_function(self, func) -> None:
        self._susceptibility = func

    def _susceptibility_function(
        self, particle: config.Particle, int_func: intf.Integrand
    ) -> float:
        kappa = getattr(particle, "kappa", 1)
        temp = particle.temperature
        if int_func.the_type == "maxwell":
            debye_length = L_Debye(particle.number_density, temp)
            xp = np.sqrt(
                1 / (2 * debye_length**2 * self.params.radar_wavenumber**2)
            )
        elif int_func.the_type == "kappa":
            debye_length = L_Debye(particle.number_density, temp, kappa=kappa)
            xp = np.sqrt(
                1 / (2 * debye_length**2 * self.params.radar_wavenumber**2)
            )
        elif int_func.the_type == "a_vdf":
            char_vel = getattr(int_func, "char_vel")
            debye_length = L_Debye(particle.number_density, temp, char_vel=char_vel)
            xp = np.sqrt(
                1 / (2 * debye_length**2 * self.params.radar_wavenumber**2)
            )
        else:
            raise ValueError("Unknown function type.")
        return xp


def L_Debye(*args, kappa=None, char_vel=None):
    """Calculate the Debye length.

    Input args may be
        n_e -- electron number density
        T_e -- electron temperature
        T_i -- ion temperature

    Returns:
        float -- the Debye length
    """
    nargin = len(args)
    if nargin == 1:
        n_e = args[0]
    elif nargin == 2:
        n_e = args[0]
        T_e = args[1]
    elif nargin == 3:
        n_e = args[0]
        T_e = args[1]
        T_i = args[2]
    else:
        raise ValueError("Invalid number of input arguments.")

    Ep0 = 1e-09 / 36 / np.pi

    if nargin < 3:
        if kappa is not None:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e**2)) * np.sqrt(
                (kappa - 3 / 2) / (kappa - 1 / 2)
            )
        elif char_vel is not None:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e**2)) * np.sqrt(
                char_vel
            )
        else:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e**2))
    else:
        LD = np.sqrt(
            Ep0 * const.k / ((max(0, n_e) / T_e + max(0, n_e) / T_i) / const.e**2)
        )

    return LD
