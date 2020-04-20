"""Constants used system wide.

The physical properties of the plasma is changed here, described below 'Input parameters'.
Only one dictionary containing plasma parameters should be uncommented at any instance in time.
"""

import os
import sys

import numpy as np
import scipy.constants as const


# Check if a test is running. Potential paths are
# ['pytest.py', 'pytest', 'test_ISR.py', '__main__.py', 'python3.7 -m unittest']
# or check if 'main.py' was used.
if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
    # DO NOT EDIT
    F_N_POINTS = 1e1
    Y_N_POINTS = 1e1
    V_N_POINTS = 1e1
else:
    F_N_POINTS = 1e3  # Number of sample points in frequency
    Y_N_POINTS = 5e2  # Number of sample points in integral variable
    V_N_POINTS = 6e4  # Number of sample points in velocity integral variable
# Adds one sample to get an even number of bins, which in
# turn give better precision in the Simpson integration.
Y_N_POINTS += 1
V_N_POINTS += 1
Y_MAX_e = 1.5e-4  # Upper limit of integration (= infinity)
Y_MAX_i = 1.5e-2
V_MAX = 3e7
ORDER = 3

# === Input parameters ===
# B -- Magnetic field strength [T]
# F0 -- Radar frequency [Hz]
# F_MAX -- Range of frequency domain [Hz]
# MI -- Ion mass in atomic mass units [u]
# NE -- Electron number density [m^(-3)]
# NU_E -- Electron collision frequency [Hz]
# NU_I -- Ion collision frequency [Hz]
# T_E -- Electron temperature [K]
# T_I -- Ion temperature [K]
# T_ES -- Temperature of suprathermal electrons in the gauss_shell VDF [K]
# THETA -- Pitch angle [1]

# For seeing gyro lines
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 2e6, 'MI': 29, 'NE': 2e10,
#        'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'THETA': 45 * np.pi / 180}
# For same plots as Hagfors
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 1.5e6, 'MI': 16, 'NE': 2e10,
#        'NU_E': 0, 'NU_I': 0, 'T_E': 1000, 'T_I': 1000, 'THETA': 0 * np.pi / 180}
# High frequency plasma lines
# I_P = {'B': 5e-4, 'F0': 933e6, 'F_MAX': 1.25e7, 'MI': 16, 'NE': 2e11, 'NU_E': 1000, 'NU_I': 0,
#        'T_E': 10000, 'T_I': 2000, 'T_ES': 90000, 'THETA': 45 * np.pi / 180}
I_P = {'B': 5e-4, 'F0': 430e6, 'F_MAX': 1e4, 'MI': 16, 'NE': 2e11, 'NU_E': 1000, 'NU_I': 0,
       'T_E': 2000, 'T_I': 1000, 'T_ES': 90000, 'THETA': 40 * np.pi / 180}

# DO NOT EDIT
K_RADAR = - 2 * I_P['F0'] * 2 * np.pi / const.c  # Radar wavenumber
f = np.linspace(- I_P['F_MAX'], I_P['F_MAX'], int(F_N_POINTS))
f = (f / I_P['F_MAX'])**3 * I_P['F_MAX']
w = 2 * np.pi * f  # Angular frequency

# Global variable used as the integrand in the Simpson integral for the spectrum
ff = None

SCALING = None
