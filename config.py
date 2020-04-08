"""Constants used system wide.
"""

import os
import sys

import numpy as np
import scipy.constants as const


# Check if a test is running
if os.path.basename(os.path.realpath(sys.argv[0])) in ['pytest.py', 'pytest']:
    # DO NOT EDIT
    F_N_POINTS = 1e1
    N_POINTS = 1e2
else:
    F_N_POINTS = 1e3  # Number of sample points in frequency
    N_POINTS = 1e5  # Number of sample points in integral variable
T_MAX_e = 1.5e-4  # Upper limit to integration (= infinity)
T_MAX_i = 1.5e-2
ORDER = 3

# === Input parameters ===
# B -- Magnetic field strength [T]
# F0 -- Radar frequency [Hz]
# F_MAX -- Range of frequency domain
# MI -- Ion mass in atomic mass units [u]
# NE -- Electron number density [m^(-3)]  (1.5e6)^2/(8.98^2)
# NU_E -- Electron collision frequency [Hz]
# NU_I -- Ion collision frequency [Hz]
# T_E -- Electron temperature [K]
# T_I -- Ion temperature [K]
# THETA -- Pitch angle [1]

# For seeing gyro lines
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 2e3, 'MI': 29, 'NE': 2e10,
#        'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'THETA': 45 * np.pi / 180}
# For same plots as Hagfors
# I_P = {'B': 35000e-9, 'F0': 430e6, 'F_MAX': 1.5e6, 'MI': 16, 'NE': 2e10,
#        'NU_E': 0, 'NU_I': 0, 'T_E': 1000, 'T_I': 1000, 'THETA': 0 * np.pi / 180}
# High frequency plasma lines
I_P = {'B': 5e-4, 'F0': 933e6, 'F_MAX': 1e7, 'MI': 16, 'NE': 2e11,
       'NU_E': 1000, 'NU_I': 0, 'T_E': 4000, 'T_I': 2000, 'THETA': 45 * np.pi / 180}

# For kappa distribution
# NOTE: kappa != n + 1/2, n in the integers
# KAPPA = 20
# NU = - KAPPA - 1 / 2

# DO NOT EDIT
K_RADAR = - 2 * I_P['F0'] * 2 * np.pi / const.c  # Radar wavenumber
f = np.linspace(- I_P['F_MAX'], I_P['F_MAX'], int(F_N_POINTS))
f = (f / I_P['F_MAX'])**3 * I_P['F_MAX']
w = 2 * np.pi * f  # Angular frequency

ff = None

# NOT USED: used for chirp-z transform
# f = np.arange(F_N_POINTS / 2) * (I_P['F_MAX'] - 0) / (F_N_POINTS / 2)  # Frequency
# dW = 2 * np.pi * (I_P['F_MAX'] - 0) / (F_N_POINTS / 2)  # Step size angular frequency
# w = np.arange(F_N_POINTS / 2) * dW  # Angular frequency

# phi = np.linspace(0, 2 * np.pi, 1000)
# a = [.9, .1, .4 * np.sin(3 * phi), .1 * np.cos(20 * phi)]
# func = 0
# for c, v in enumerate(a):
#     func += v * abs(np.cos(phi))**(c + 1)
# # plt.figure()
# _, axs = plt.subplots()
# axs.set_aspect('equal', 'box')
# plt.plot(func * np.cos(phi), func * np.sin(phi))
# plt.show()
