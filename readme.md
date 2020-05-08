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
Makes plots of an incoherent scatter radar spectrum based on the theory presented in [Hagfors (1961)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699) and [Mace (2003)](https://aip.scitation.org/doi/pdf/10.1063/1.1570828).

## Installing <a name = "installing"></a>
The program is built using `python3.8` and tested on **macOS**.

Run
```
pip install -r requirements.txt
```
from directory `isr_spectrum` where `requirements.txt` is located to install all needed packages, and then navigate to the `program` folder
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
Z -- Height of real data [100, 599] [km]
mat_file -- Important when using real data and decides the time of day
```
which are given in `main.py`.

### Calculation method
The program support different methods of calculating the spectrum, based on how you assume the particles to be distributed. This includes a Maxwellian distribution and a kappa distribution, in addition to any other arbitrary isotropic distribution.

The version that determine the calculation method is given in `main.py`, with additional keyword arguments that decide how to plot the result from the calculation.

## File structure <a name = "structure"></a>
```
isr_spectrum/
├── extra/
│   ├── simple_calculations.py
│   └── simple_plots.py
├── program/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── Arecibo-photo-electrons/
│   │   │   ├── E4fe.dat
│   │   │   ├── fe_zmuE-01.mat
│   │   │   ├── fe_zmuE-01.mat-01.png
│   │   │   ├── ...
│   │   │   ├── fe_zmuE-15.mat
│   │   │   ├── fe_zmuE-15.mat-01.png
│   │   │   ├── SzeN.dat
│   │   │   ├── theta_lims.dat
│   │   │   ├── timeOfDayUT.dat
│   │   │   └── z4fe.dat
│   │   └── read.py
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
│       ├── parallel/
│       │   ├── __init__.py
│       │   ├── parallelization.py
│       │   └── v_int_parallel.py
│       ├── spectrum_calculation.py
│       └── vdfs.py
├── readme.md
└── requirements.txt
```
