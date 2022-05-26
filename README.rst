inscar
======

    INcoherent SCAtter Radar spectrum

|PyPI| |Status| |Python Version| |License| |Read the Docs| |Tests| |Codecov| |DOI|
|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/inscar.svg
   :target: https://pypi.org/project/inscar/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/inscar.svg
   :target: https://pypi.org/project/inscar/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/inscar
   :target: https://pypi.org/project/inscar
   :alt: Python Version
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/inscar/latest.svg?label=Read%20the%20Docs
   :target: https://inscar.readthedocs.io/
   :alt: Read the documentation at https://ncdump-rich.readthedocs.io/
.. |Tests| image:: https://github.com/engeir/inscar/workflows/Tests/badge.svg
   :target: https://github.com/engeir/inscar/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/engeir/inscar/branch/master/graph/badge.svg?token=P8S18UILSB
   :target: https://codecov.io/gh/engeir/inscar
   :alt: Codecov
.. |DOI| image:: https://zenodo.org/badge/233043566.svg
   :target: https://zenodo.org/badge/latestdoi/233043566
   :alt: pre-commit
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

.. [![release](https://img.shields.io/github/release/engeir/inscar.svg)](https://github.com/engeir/inscar/releases/latest)
.. ![CodeQL](https://github.com/engeir/inscar/workflows/CodeQL/badge.svg)

.. image:: ./img/normal_is_spectra.png

Info
----

Calculates an incoherent scatter radar spectrum based on the theory presented in
`Hagfors (1961)`_ and `Mace (2003)`_.

Installing
----------

You can install *inscar* via pip_ from PyPI_:

.. code:: console

   $ pip install inscar

Usage
-----

Please see the `Modules Reference <Modules_>`_ for details.

Numba
^^^^^

Faster integration is accomplished by computing in parallel. This is
accomplished using `numba`.

Physical environment
^^^^^^^^^^^^^^^^^^^^

The available plasma parameters that can be changed within the program are

.. code:: text

    === Input parameters ===
    K_RADAR -- Radar wavenumber (= -4pi(radar frequency)/(speed of light)) [m^(-1)]
    B -- Magnetic field strength [T]
    MI -- Ion mass in atomic mass units [u]
    NE -- Electron number density [m^(-3)]
    NU_E -- Electron collision frequency [Hz]
    NU_I -- Ion collision frequency [Hz]
    T_E -- Electron temperature [K]
    T_I -- Ion temperature [K]
    T_ES -- Temperature of suprathermal electrons in the gauss_shell VDF [K] (no longer in use)
    THETA -- Aspect angle [1]
    Z -- Height used for calculated distribution [100, 599] [km]
    mat_file -- Time of day for calculated distribution
    pitch_angle -- Pitch angle for calculated distribution

Calculation method
^^^^^^^^^^^^^^^^^^

The program support different methods of calculating the spectrum, based on how you
assume the particles to be distributed. This includes a Maxwellian distribution and a
kappa distribution, in addition to any arbitrary isotropic distribution.

.. _Hagfors (1961): https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699
.. _Mace (2003): https://aip.scitation.org/doi/pdf/10.1063/1.1570828
.. _PyPI: https://pypi.org/
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Modules: https://inscar.readthedocs.io/en/latest/modules.html
