"""Analytical accuracy checks for quadrature and Maxwellian spectra."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import constants as const
from scipy.integrate import simpson
from scipy.special import wofz

import inscar as isr
from inscar import numba_integration as nb_int
from inscar.spectrum_calculation import get_debye_length


def test_debye_length_uses_consistent_si_constants():
    """Check the Debye length against its defining SI relation."""
    density, temperature = 1e11, 4000.0
    length = get_debye_length(density, temperature)
    assert_allclose(
        density * const.e**2 * length**2 / (const.k * temperature),
        const.epsilon_0,
        rtol=1e-14,
        atol=0,
    )


def test_simpson_nonuniform_complex_quadratic():
    """A quadratic has an exact integral even on a nonuniform grid."""
    x = np.linspace(0, 2, 101) ** 3
    y = (2 + 3j) * x**2 + (1 - 2j) * x + 4j
    expected = (2 + 3j) * x[-1] ** 3 / 3 + (1 - 2j) * x[-1] ** 2 / 2 + 4j * x[-1]
    assert_allclose(nb_int.simpson(y, x), expected, rtol=1e-14)


def test_simpson_matches_scipy():
    """Compare the compiled quadrature to SciPy on a complex integrand."""
    x = np.linspace(0, 2, 801) ** 3
    y = np.exp(-(x**2) - 7j * x)
    assert_allclose(nb_int.simpson(y, x), simpson(y, x=x), rtol=1e-13, atol=1e-15)


def test_simpson_gordeyev_gaussian():
    """Check a Fourier Gaussian integral against the Faddeeva function."""
    x = np.linspace(0, 2, 4001) ** 3
    w = np.array([0.0, 0.5, 2.0, 5.0, 12.0])
    result = nb_int.inner_int_simpson(w, x, np.exp(-(x**2)))
    expected = np.sqrt(np.pi) / 2 * wofz(-w / 2)
    assert_allclose(result, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_simpson_rejects_invalid_sizes(size):
    """Reject grids that cannot form pairs of intervals."""
    with pytest.raises(ValueError, match="odd lengths"):
        nb_int.simpson(np.ones(size), np.arange(size, dtype=float))


def test_gordeyev_method_configuration():
    """Validate selection at construction and assignment without changing default."""
    particle = isr.Particle()
    assert particle.gordeyev_method == "trapezoid"
    particle.gordeyev_method = "simpson"
    with pytest.raises(ValueError):
        particle.gordeyev_method = "unknown"
    with pytest.raises(ValueError):
        isr.Particle(gordeyev_method="unknown")


def test_maxwellian_plasma_line():
    """Resolve a narrow plasma line and compare with an analytic noise response."""
    density, ti, te = 1e11, 1000.0, 4000.0
    params = isr.Parameters(
        radar_frequency=430e6,
        magnetic_field_strength=40e-6,
        aspect_angle=0,
        frequency_range=(3e6, 3.3e6),
        frequency_size=121,
    )
    k = abs(params.radar_wavenumber)
    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    responses = []
    for label, mass, temperature in [
        ("ion", 30 * const.atomic_mass, ti),
        ("electron", const.m_e, te),
    ]:
        particle = isr.Particle(
            mass=mass,
            temperature=temperature,
            number_density=density,
            gordeyev_size=64001,
            gordeyev_upper_lim=15 / (k * np.sqrt(const.k * temperature / mass)),
            gordeyev_method="simpson",
        )
        getattr(sim, "set_" + label)(particle)
        getattr(sim, "set_" + label + "_integration_function")(isr.IntMaxwell())
        thermal_speed = np.sqrt(2 * const.k * temperature / mass)
        z = params.angular_frequency / (k * thermal_speed)
        debye_squared = const.epsilon_0 * const.k * temperature / (density * const.e**2)
        susceptibility = (1 + z * 1j * np.sqrt(np.pi) * wofz(z)) / (
            k**2 * debye_squared
        )
        noise = np.sqrt(np.pi) / (k * thermal_speed) * np.exp(-(z**2))
        responses.append((susceptibility, noise))
    (chi_i, noise_i), (chi_e, noise_e) = responses
    dielectric = 1 + chi_e + chi_i
    expected = 2 * (
        abs(chi_e / dielectric) ** 2 * noise_i
        + abs((1 + chi_i) / dielectric) ** 2 * noise_e
    )
    _, actual = sim.calculate_spectrum()
    actual *= 2 * np.pi / density
    tolerance = 1e-6
    assert np.max(abs(actual - expected)) / np.max(expected) < tolerance
