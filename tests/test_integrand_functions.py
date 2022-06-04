"""Test cases for the integrand functions module."""

import inspect

import inscar
from inscar import integrand_functions as ins


def test_contents():
    """Test that all classes contain needed methods and attributes."""
    params = inscar.Parameters(frequency_size=1e3 + 1)
    ion = inscar.Particle(gordeyev_size=1e3 + 1, velocity_size=1e3 + 1)
    for n, c in inspect.getmembers(ins, inspect.isclass):
        if c.__module__ == "inscar.integrand_functions" and n != "Integrand":
            o = c()
            assert issubclass(c, ins.Integrand)
            assert hasattr(o, "the_type")
            o.initialize(params, ion)
            o.integrand()
