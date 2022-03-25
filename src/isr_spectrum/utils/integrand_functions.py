"""Script containing the integrands used in the Gordeyev integral.
"""

from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

from isr_spectrum.utils import vdfs
from isr_spectrum.utils.njit import gordeyev_njit
from isr_spectrum.utils import config


class Integrand(ABC):
    """Base class for an integrand object.

    Arguments:
        ABC {ABC} -- abstract base class
    """

    @abstractproperty
    def the_type(self) -> str:
        """The type of the intregrand implementation."""

    @abstractmethod
    def initialize(self, params: config.Parameters, particle: config.Particle):
        """Needs an initialization method.

        Arguments:
            y {np.ndarray} -- array for integration variable
            params {dict} -- dictionary holding all needed parameters
        """

    @abstractmethod
    def integrand(self) -> np.ndarray:
        """Method that returns the np.ndarray that is used as the integrand."""


class IntKappa(Integrand):
    """Implementation of the integrand of the Gordeyev
    integral for the kappa distribution from Mace (2003).

    Arguments:
        Integrand {ABC} -- base class used to create integrand objects
    """

    the_type = "kappa"

    def __init__(self):
        self.params: config.Parameters
        self.particle: config.Particle
        self.gyro_frequency: float
        self.Z: float
        self.Kn: float

    def initialize(self, params: config.Parameters, particle: config.Particle):
        self.params = params
        self.particle = particle
        self.gyro_frequency = (
            const.e * self.params.magnetic_field_strength / self.particle.mass
        )
        self.z_func()

    def z_func(self):
        y = self.particle.gordeyev_axis
        theta_2 = (
            2
            * ((self.particle.kappa - 3 / 2) / self.particle.kappa)
            * self.particle.temperature
            * const.k
            / self.particle.mass
        )
        self.Z = (2 * self.particle.kappa) ** (1 / 2) * (
            self.params.radar_wavenumber ** 2
            * np.sin(self.params.aspect_angle) ** 2
            * theta_2
            / self.gyro_frequency ** 2
            * (1 - np.cos(self.gyro_frequency * y))
            + 1
            / 2
            * self.params.radar_wavenumber ** 2
            * np.cos(self.params.aspect_angle) ** 2
            * theta_2
            * y ** 2
        ) ** (1 / 2)
        self.Kn = sps.kv(self.particle.kappa + 1 / 2, self.Z)
        self.Kn[self.Kn == np.inf] = 1

    def integrand(self):
        y = self.particle.gordeyev_axis
        return (
            self.Z ** (self.particle.kappa + 0.5)
            * self.Kn
            * np.exp(-y * self.particle.collision_frequency)
        )


class IntMaxwell(Integrand):
    """Implementation of the intregrand in the Gordeyev
    integral for the Maxwellian distribution from
    e.g. Hagfors (1961) or Mace (2003).

    Arguments:
        Integrand {ABC} -- base class used to create integrand objects
    """

    the_type = "maxwell"

    def __init__(self):
        self.params: config.Parameters
        self.particle: config.Particle
        self.gyro_frequency: float

    def initialize(self, params: config.Parameters, particle: config.Particle):
        self.params = params
        self.particle = particle
        self.gyro_frequency = (
            const.e * self.params.magnetic_field_strength / self.particle.mass
        )

    def integrand(self):
        return np.exp(
            -self.particle.gordeyev_axis * self.particle.collision_frequency
            - self.params.radar_wavenumber ** 2
            * np.sin(self.params.aspect_angle) ** 2
            * self.particle.temperature
            * const.k
            / (self.particle.mass * self.gyro_frequency ** 2)
            * (1 - np.cos(self.gyro_frequency * self.particle.gordeyev_axis))
            - 0.5
            * (
                self.params.radar_wavenumber
                * np.cos(self.params.aspect_angle)
                * self.particle.gordeyev_axis
            )
            ** 2
            * self.particle.temperature
            * const.k
            / self.particle.mass
        )


class IntLong(Integrand):
    """Implementation of the integrand in the Gordeyev
    integral for the isotropic distribution from Mace (2003).

    Arguments:
        Integrand {ABC} -- base class used to create integrand objects
    """

    the_type = "a_vdf"

    def __init__(self):
        self.params: config.Parameters
        self.particle: config.Particle
        self.char_vel: float
        self.vdf = vdfs.VdfMaxwell
        self.gyro_frequency: float

    def set_vdf(self, vdf):
        self.vdf = vdf

    def initialize(self, params: config.Parameters, particle: config.Particle):
        self.params = params
        self.particle = particle
        self.gyro_frequency = (
            const.e * self.params.magnetic_field_strength / self.particle.mass
        )

    def v_int(self):
        v = self.particle.velocity_axis
        y = self.particle.gordeyev_axis
        f = self.vdf(self.params, self.particle)
        # if self.params["vdf"] == "kappa":
        #     f = vdfs.VdfKappa(v, self.params)
        # elif self.params["vdf"] == "kappa_vol2":
        #     f = vdfs.VdfKappa2(v, self.params)
        # elif self.params["vdf"] == "gauss_shell":
        #     f = vdfs.VdfGaussShell(v, self.params)
        # elif self.params["vdf"] == "real_data":
        #     f = vdfs.VdfRealData(v, self.params)
        # else:  # self.params["vdf"] == "maxwell":
        #     f = vdfs.VdfMaxwell(v, self.params)

        # Compare the velocity integral to the Maxwellian case.
        # This way we make up for the change in characteristic velocity
        # and Debye length for different particle distributions.
        res_maxwell = gordeyev_njit.integrate_velocity(
            self.particle.gordeyev_axis,
            v,
            vdfs.VdfMaxwell(self.params, self.particle).f_0(),
            self.params.radar_wavenumber,
            self.params.aspect_angle,
            self.gyro_frequency,
        )
        int_maxwell = si.simps(res_maxwell, y)
        v_func = f.f_0()
        res = gordeyev_njit.integrate_velocity(
            self.particle.gordeyev_axis,
            v,
            v_func,
            self.params.radar_wavenumber,
            self.params.aspect_angle,
            self.gyro_frequency,
        )
        int_res = si.simps(res, y)
        # The scaling of the factor describing the characteristic velocity
        self.char_vel = int_maxwell / int_res
        print(
            f"Debye length of the current distribution is {self.char_vel}"
            + " times the Maxwellian Debye length."
        )
        return res

    def p_d(self):
        y = self.particle.gordeyev_axis
        # At $ y=0 $ we get $ 0/0 $, so we use
        # $ \lim_{y\rightarrow 0^+}\mathrm{d}p/\mathrm{d}y = |k| |w_c| / \sqrt(w_c^2) $ (from above, opposite sign from below)
        cos_t = np.cos(self.params.aspect_angle)
        sin_t = np.sin(self.params.aspect_angle)
        w_c = self.gyro_frequency
        num = (
            abs(self.params.radar_wavenumber)
            * abs(w_c)
            * (cos_t ** 2 * w_c * y + sin_t ** 2 * np.sin(w_c * y))
        )
        term1 = (cos_t * w_c * y) ** 2
        term2 = -2 * sin_t ** 2 * np.cos(w_c * y)
        term3 = 2 * sin_t ** 2
        den = w_c * (term1 + term2 + term3) ** 0.5
        # np.sign(y[-1]) takes care of weather the limit should be considered taken from above or below.
        # The last element of the np.ndarray is chosen since it is assumed y runs from 0 to some finite real number.
        first = np.sign(y[-1]) * abs(self.params.radar_wavenumber) * abs(w_c) / abs(w_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out[np.where(den == 0.0)[0]] = first

        return out

    def integrand(self):
        return self.p_d() * self.v_int()
