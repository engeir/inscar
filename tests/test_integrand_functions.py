"""Test cases for the integrand functions module."""

import inspect

from inscar import integrand_functions as ins


def test_contents():
    """Test that all classes contain needed methods and attributes."""
    for n, c in inspect.getmembers(ins, inspect.isclass):
        if c.__module__ == "inscar.integrand_functions" and n != "Integrand":
            assert issubclass(c, ins.Integrand)
            assert hasattr(c(), "the_type")
            initialize = getattr(c(), "initialize", None)
            assert initialize is not None
            integrand = getattr(c(), "integrand", None)
            assert integrand is not None
