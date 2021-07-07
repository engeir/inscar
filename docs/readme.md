[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![release](https://img.shields.io/github/release/engeir/isr_spectrum.svg)](https://github.com/engeir/isr_spectrum/releases/latest)
[![DOI](https://zenodo.org/badge/233043566.svg)](https://zenodo.org/badge/latestdoi/233043566)
![ISR spectrum](https://github.com/engeir/isr_spectrum/workflows/ISR%20spectrum/badge.svg)
![CodeQL](https://github.com/engeir/isr_spectrum/workflows/CodeQL/badge.svg)

## Contents
- [Info](#info)
- [Installing](#installing)
- [Usage](#usage)
- [File structure](#structure)

## Info <a name = "info"></a>
Calculates an incoherent scatter radar spectrum based on the theory presented in [Hagfors
(1961)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699) and [Mace (2003)](https://aip.scitation.org/doi/pdf/10.1063/1.1570828).

## Installing <a name = "installing"></a>

The program is built using `python3.9.2` and tested on **macOS** and **ubuntu20.10**.

Clone the repository using `git clone https://github.com/engeir/isr-spectrum.git` or
download the latest release,
[v2.0.0](https://github.com/engeir/isr-spectrum/archive/refs/tags/v2.0.0.zip). Dependencies are handled
by [`poetry`](https://python-poetry.org/), which can be installed with

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

To install using `poetry`, run
```sh
poetry install
```
or install using the provided `requirements.txt` file:
```sh
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

### TL;DR

Start the program with command
```sh
python src/isr_spectrum/main.py
```

### Numba

Faster integration is accomplished by computing in parallel. By default this is
accomplished using the `multiprocessing` module, but a faster implementation using `numba`
is available. To use the `numba` implementation, set `NJIT = True` in
[config.py](src/isr_spectrum/inputs/config.py).

### Physical environment

The available plasma parameters that can be changed within the program are
```
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
```
which are given in [`main.py`](src/isr_spectrum/main.py).

### Calculation method

The program support different methods of calculating the spectrum, based on how you assume
the particles to be distributed. This includes a Maxwellian distribution and a kappa
distribution, in addition to any arbitrary isotropic distribution.

The version that determine the calculation method is described in
[`spectrum_calculation.py`](src/isr_spectrum/utils/spectrum_calculation.py), in the docstring of
the function [`isr_spectrum`](src/isr_spectrum/utils/spectrum_calculation.py#L18), with additional
keyword arguments.

## File structure <a name = "structure"></a>

Output from `tree --dirsfirst -I __pycache__`:

```
isr-spectrum/
├── docs
│   ├── img
│   │   └── normal_is_spectra.png
│   ├── _layouts
│   │   └── index.html
│   ├── _config.yml
│   └── readme.md
├── src
│   ├── isr_spectrum
│   │   ├── data
│   │   │   └── arecibo
│   │   │       ├── E4fe.dat
│   │   │       ├── fe_zmuE-07.mat
│   │   │       ├── SzeN.dat
│   │   │       ├── theta_lims.dat
│   │   │       ├── timeOfDayUT.dat
│   │   │       └── z4fe.dat
│   │   ├── inputs
│   │   │   ├── config.py
│   │   │   └── __init__.py
│   │   ├── plotting
│   │   │   ├── hello_kitty.py
│   │   │   ├── __init__.py
│   │   │   ├── plot_class.py
│   │   │   └── reproduce.py
│   │   ├── utils
│   │   │   ├── njit
│   │   │   │   └── gordeyev_njit.py
│   │   │   ├── parallel
│   │   │   │   ├── gordeyev_int_parallel.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── v_int_parallel.py
│   │   │   ├── __init__.py
│   │   │   ├── integrand_functions.py
│   │   │   ├── read.py
│   │   │   ├── spectrum_calculation.py
│   │   │   └── vdfs.py
│   │   └── main.py
│   └── __init__.py
├── tests
│   ├── __init__.py
│   └── test_ISR.py
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── readme.md
├── requirements.txt
└── setup.cfg

13 directories, 35 files
```
