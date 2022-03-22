"""Script containing the integrands used in the Gordeyev integral.
"""

from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

from isr_spectrum.utils import vdfs
from isr_spectrum.utils.njit import gordeyev_njit


class Integrand(ABC):
    """Base class for an integrand object.

    Arguments:
        ABC {ABC} -- abstract base class
    """

    @abstractproperty
    def the_type(self) -> str:
        """The type of the intregrand implementation."""

    @abstractmethod
    def initialize(self, y, params):
        """Needs an initialization method.

        Arguments:
            y {np.ndarray} -- array for integration variable
            params {dict} -- dictionary holding all needed parameters
        """

    @abstractmethod
    def integrand(self):
        """Method that returns the np.ndarray that is used as the integrand."""


class IntKappa(Integrand):
    """Implementation of the integrand of the Gordeyev
    integral for the kappa distribution from Mace (2003).

    Arguments:
        Integrand {ABC} -- base class used to create integrand objects
    """

    the_type = "kappa"

    def __init__(self):
        self.y = np.array([])
        self.params = {}
        self.Z = float
        self.Kn = float

    def initialize(self, y, params):
        self.y = y
        self.params = params
        self.z_func()

    def z_func(self):
        theta_2 = (
            2
            * ((self.params["kappa"] - 3 / 2) / self.params["kappa"])
            * self.params["T"]
            * const.k
            / self.params["m"]
        )
        self.Z = (2 * self.params["kappa"]) ** (1 / 2) * (
            self.params["K_RADAR"] ** 2
            * np.sin(self.params["THETA"]) ** 2
            * theta_2
            / self.params["w_c"] ** 2
            * (1 - np.cos(self.params["w_c"] * self.y))
            + 1
            / 2
            * self.params["K_RADAR"] ** 2
            * np.cos(self.params["THETA"]) ** 2
            * theta_2
            * self.y ** 2
        ) ** (1 / 2)
        self.Kn = sps.kv(self.params["kappa"] + 1 / 2, self.Z)
        self.Kn[self.Kn == np.inf] = 1

    def integrand(self):
        return (
            self.Z ** (self.params["kappa"] + 0.5)
            * self.Kn
            * np.exp(-self.y * self.params["nu"])
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
        self.y = np.array([])
        self.params = {}

    def initialize(self, y, params):
        self.y = y
        self.params = params

    def integrand(self):
        return np.exp(
            -self.y * self.params["nu"]
            - self.params["K_RADAR"] ** 2
            * np.sin(self.params["THETA"]) ** 2
            * self.params["T"]
            * const.k
            / (self.params["m"] * self.params["w_c"] ** 2)
            * (1 - np.cos(self.params["w_c"] * self.y))
            - 0.5
            * (self.params["K_RADAR"] * np.cos(self.params["THETA"]) * self.y) ** 2
            * self.params["T"]
            * const.k
            / self.params["m"]
        )


class IntLong(Integrand):
    """Implementation of the intregrand in the Gordeyev
    integral for the isotropic distribution from Mace (2003).

    Arguments:
        Integrand {ABC} -- base class used to create integrand objects
    """

    the_type = "a_vdf"

    def __init__(self):
        self.y = np.array([])
        self.params = {}
        self.char_vel = float
        self.vdf = vdfs.VdfMaxwell

    def set_vdf(self, vdf):
        self.vdf = vdf

    def initialize(self, y, params):
        self.y = y
        self.params = params

    def v_int(self):
        v = np.linspace(0, cf.V_MAX ** (1 / cf.ORDER), int(cf.V_N_POINTS)) ** cf.ORDER
        # f = self.vdf(v, self.params)
        if self.params["vdf"] == "kappa":
            f = vdfs.VdfKappa(v, self.params)
        elif self.params["vdf"] == "kappa_vol2":
            f = vdfs.VdfKappa2(v, self.params)
        elif self.params["vdf"] == "gauss_shell":
            f = vdfs.VdfGaussShell(v, self.params)
        elif self.params["vdf"] == "real_data":
            f = vdfs.VdfRealData(v, self.params)
        else:  # self.params["vdf"] == "maxwell":
            f = vdfs.VdfMaxwell(v, self.params)

        # Compare the velocity integral to the Maxwellian case.
        # This way we make up for the change in characteristic velocity
        # and Debye length for different particle distributions.
        res_maxwell = v_int_parallel.integrand(
            self.y, self.params, v, vdfs.VdfMaxwell(v, self.params).f_0()
        )
        int_maxwell = si.simps(res_maxwell, self.y)
        if cf.NJIT:
            v_func = f.f_0()
            res = gordeyev_njit.integrate_velocity(
                self.y,
                v,
                v_func,
                self.params["K_RADAR"],
                self.params["THETA"],
                self.params["w_c"],
            )
        else:
            res = v_int_parallel.integrand(self.y, self.params, v, f.f_0())
        int_res = si.simps(res, self.y)
        # The scaling of the factor describing the characteristic velocity
        self.char_vel = int_maxwell / int_res
        print(
            f"Debye length of the current distribution is {self.char_vel}"
            + "times the Maxwellian Debye length."
        )
        return res

    def p_d(self):
        # At $ y=0 $ we get $ 0/0 $, so we use
        # $ \lim_{y\rightarrow 0^+}\mathrm{d}p/\mathrm{d}y = |k| |w_c| / \sqrt(w_c^2) $ (from above, opposite sign from below)
        cos_t = np.cos(self.params["THETA"])
        sin_t = np.sin(self.params["THETA"])
        w_c = self.params["w_c"]
        num = (
            abs(self.params["K_RADAR"])
            * abs(w_c)
            * (cos_t ** 2 * w_c * self.y + sin_t ** 2 * np.sin(w_c * self.y))
        )
        term1 = (cos_t * w_c * self.y) ** 2
        term2 = -2 * sin_t ** 2 * np.cos(w_c * self.y)
        term3 = 2 * sin_t ** 2
        den = w_c * (term1 + term2 + term3) ** 0.5
        # np.sign(y[-1]) takes care of weather the limit should be considered taken from above or below.
        # The last element of the np.ndarray is chosen since it is assumed y runs from 0 to some finite real number.
        first = np.sign(self.y[-1]) * abs(self.params["K_RADAR"]) * abs(w_c) / abs(w_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out[np.where(den == 0.0)[0]] = first

        return out

    def integrand(self):
        return self.p_d() * self.v_int()
