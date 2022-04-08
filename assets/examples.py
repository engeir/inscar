"""Example usage of the library."""

from typing import Union

import attr
import matplotlib.pyplot as plt
import numpy as np
import read
import scipy.constants as const
import scipy.integrate as si

import isr_spectrum as isr
import isr_spectrum.utils.vdfs as vdfs


@attr.s
class RealDataParticle(isr.Particle):
    mat_file: str = attr.ib(default="fe_zmuE-07.mat")
    pitch_angle: Union[int, str] = attr.ib(default="all")
    z: int = attr.ib(default=300)


class VdfRealData(vdfs.Vdf):
    """Create an object that make distribution functions from
    a 1D array.

    Arguments:
        VDF {ABC} -- abstract base class to make VDF objects
    """

    def __init__(self, params: isr.Parameters, particle: RealDataParticle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params: isr.Parameters
            Parameters object.
        particle: isr.particle
            Particle object.
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self):
        v = self.particle.velocity_axis
        func = read.interpolate_data(self.particle)
        f = func * v**2 * 4 * np.pi
        self.A = 1 / si.simps(f, v)

    def f_0(self):
        return self.A * read.interpolate_data(self.particle)


def info():
    p = isr.Parameters(aspect_angle=45)
    print(p.aspect_angle)
    print(p.radar_wavenumber)
    p.radar_frequency = 430e5
    print(p.radar_wavenumber)
    p.aspect_angle = 360.5
    print(p.aspect_angle)


def ion_line():
    e = isr.Particle(temperature=200, kappa=8)
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()
    plt.plot(x, y)
    # plt.show()


def ion_line_long():
    e = isr.Particle(temperature=200, kappa=8)
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntLong())
    x, y = sim.calculate_spectrum()
    plt.plot(x, y)
    plt.show()


def plasma_line():
    e = isr.Particle(temperature=5000, kappa=8)
    m_i = 16 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=2000,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(3.5e6, 7e6),
        frequency_size=int(1e3 + 1),
        radar_frequency=933e6,
        magnetic_field_strength=50000e-9,
        aspect_angle=0,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()
    plt.plot(x, y)
    plt.show()


def gyro_line():
    e = isr.Particle(temperature=200, kappa=8, number_density=2e10)
    m_i = 16 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-2e6, 2e6),
        frequency_size=int(1e4 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=3.5e-5,
        aspect_angle=(360 + 135),
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()
    plt.semilogy(x, y)
    plt.show()


def real_data_custom_vdf():
    e = RealDataParticle(
        temperature=200, kappa=8, velocity_size=10001, gordeyev_size=10001
    )
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
        gordeyev_size=10001,
        velocity_size=10001,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    electron_int_func = isr.IntLong()
    electron_int_func.set_vdf(VdfRealData)
    sim.set_electron_integration_function(electron_int_func)
    x, y = sim.calculate_spectrum()
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    info()
    ion_line()
    ion_line_long()
    plasma_line()
    gyro_line()
    real_data_custom_vdf()
