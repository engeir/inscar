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
    Y_N_POINTS = 6e4  # Number of sample points in integral variable
    V_N_POINTS = 1e4  # Number of sample points in velocity integral variable
# Adds one sample to get an even number of bins, which in
# turn give better precision in the Simpson integration.
Y_N_POINTS += 1
V_N_POINTS += 1
Y_MAX_e = 1.5e-4  # Upper limit of integration (= infinity)
Y_MAX_i = 1.5e-2
# When using real data, E_max = 110 eV -> 6.22e6 m/s
# V_MAX = 3e7
V_MAX = 6e6
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
# Z -- Height of real data [100, 599] [km]
# mat_file -- Important when using real data and decides the time of day

RIDGES = 15
el_temp = []
e_t_0 = 2000
e_t = 1000
for i in range(RIDGES):
    el_temp.append(e_t_0 + e_t * i)
heights = []
height_0 = 150
height = 50
for i in range(RIDGES):
    heights.append(height_0 + height * i)
mat_file = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# for i in range(RIDGES):
#     mat_file.append(i)
for m_i, m_v in enumerate(mat_file):
    if m_v in range(1, 10):
        mat_file[m_i] = f'fe_zmuE-0{m_v}.mat'
    elif m_v in range(10, 16):
        mat_file[m_i] = f'fe_zmuE-{m_v}.mat'
# Rough estimate of n_e(height) based on plot.
n_e = [4e9, 5e9, 1e10, 3e10, 8e10]

# For seeing gyro lines
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 2e6, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0,
#        'T_E': 200, 'T_I': 200, 'T_ES': 10000, 'THETA': 45 * np.pi / 180}
# For same plots as Hagfors
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 1.5e6, 'MI': 16, 'NE': 2e10,
#        'NU_E': 0, 'NU_I': 0, 'T_E': 1000, 'T_I': 1000, 'THETA': 0 * np.pi / 180}
# High frequency plasma lines
I_P = {'B': 5e-4, 'F0': 933e6, 'F_MAX': 8e6, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0,
       'T_E': 5000, 'T_I': 2000, 'T_ES': 90000, 'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': mat_file}
# I_P = {'B': 5e-4, 'F0': 430e6, 'F_MAX': 8e6, 'MI': 16, 'NE': 2e10, 'NU_E': 1000, 'NU_I': 0,
#        'T_E': 2000, 'T_I': 1000, 'T_ES': 90000, 'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': mat_file}

# DO NOT EDIT
K_RADAR = - 2 * I_P['F0'] * 2 * np.pi / const.c  # Radar wavenumber
# If 'plasma' == True, might as well set f_min ≈ 1e6
f = np.linspace(1e4, I_P['F_MAX'], int(F_N_POINTS))
f = (f / I_P['F_MAX'])**1 * I_P['F_MAX']
w = 2 * np.pi * f  # Angular frequency
