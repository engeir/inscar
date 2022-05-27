"""INcoherent SCAtter Radar spectrum."""

from importlib_metadata import version

from inscar.config import *  # noqa: F401, F403
from inscar.integrand_functions import *  # noqa: F401, F403
from inscar.spectrum_calculation import SpectrumCalculation  # noqa: F401
from inscar.vdfs import *  # noqa: F401, F403

__version__ = version(__package__)
