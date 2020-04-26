<h1 align="center">ISR spectrum</h1>
<div align="center">

![ISR spectrum](https://github.com/engeir/code-for-master/workflows/ISR%20spectrum/badge.svg)

</div>

---

## Contents
- [Info](#info)
- [Installing](#installing)
- [Usage](#usage)
- [File structure](#structure)

## Info <a name = "info"></a>
Makes plots of an incoherent scatter radar spectrum based on the theory presented in [Hagfors (1961)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699).

## Installing <a name = "installing"></a>
The program is built using `python3.8` and tested on **macOS**.

Before trying the program, run
```
pip install -r requirements.txt
```
to install all needed packages, and then navigate to the `program` folder
```
cd program
```
Start the program with command
```
python3 main.py
```

## Usage <a name = "usage"></a>
### Physical environment
The available plasma parameters that can be changed within the program are
```
=== Input parameters ===
B -- Magnetic field strength [T]
F0 -- Radar frequency [Hz]
F_MAX -- Range of frequency domain [Hz]
MI -- Ion mass in atomic mass units [u]
NE -- Electron number density [m^(-3)]
NU_E -- Electron collision frequency [Hz]
NU_I -- Ion collision frequency [Hz]
T_E -- Electron temperature [K]
T_I -- Ion temperature [K]
T_ES -- Temperature of suprathermal electrons in the gauss_shell VDF [K]
THETA -- Pitch angle [1]
```
which are found in `program/inputs/config.py`.

### Calculation method
The program support different methods of calculating the spectrum, based on how you assume the particles to be distributed. This includes a Maxwellian distribution and a kappa distribution, in addition to any other arbitrary isotropic distribution.

The version that determine the calculation method is given in `main.py`, with additional keyword arguments that decide how to plot the result from the calculation.

## File structure <a name = "structure"></a>
```
isr_spectrum/
├── extra/
│   ├── simple_calculations.py
│   └── simple_plots.py
├── not_in_use/
│   ├── chirpz.py
│   ├── int_cy.pyx
│   ├── profile.py
│   ├── pure_cython.pyx
│   └── setup.py
├── program/
│   ├── inputs/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── main.py
│   ├── test/
│   │   ├── __init__.py
│   │   └── test_ISR.py
│   └── utils/
│       ├── __init__.py
│       ├── integrand_functions.py
│       ├── parallelization.py
│       ├── spectrum_calculation.py
│       ├── v_int_parallel.py
│       └── vdfs.py
├── readme.md
└── requirements.txt
```
