"""Constants used system wide.
"""

import numpy as np

C_0 = 299792458  # Speed of light
M_E = 9.10938291e-31  # electron rest mass [kg]
M_P = 1.672621778e-27  # proton rest mass [kg]
M_N = 1.674927352e-27  # neutron rest mass [kg]
Q_E = 1.602176565e-19  # elementary charge [C]

K_B = 1.380662e-23  # Boltzmann constant [J/K]
MY_0 = 4 * np.pi * 1e-7  # Permeability [Vs/Am]

N_POINTS = 1e7
T_MAX = 1e3

# === Input parameters ===
B = 35000e-9  # Magnetic field strength [T]
F0 = 430e6  # Radar frequency [Hz]
F_ION_MAX = 3e6  # F0 / F_ION_MAX > 15 000
# F_ION = np.linspace(0, F_ION_MAX, 1e6)  # Ion frequency [Hz]
# F_ION += 1
MI = 16  # Ion mass in atomic mass units [u]
NE = 2e10  # 2e11  # Electron number density [m^(-3)]  (1.5e6)^2/(8.98^2)
NU_E = 0  # Electron collision frequency [Hz]
NU_I = 0  # Ion collision frequency [Hz]
T_E = 200  # Electron temperature [K]
T_I = 200  # Ion temperature [K]
THETA = 45 * np.pi / 180  # Pitch angle

# DO NOT EDIT
K_RADAR = F0 * 2 * np.pi / C_0  # Radar wavenumber
f = np.arange(N_POINTS / 2) * (F_ION_MAX - 0) / (N_POINTS / 2)  # Frequency
dW = 2 * np.pi * (F_ION_MAX - 0) / (N_POINTS / 2)  # Step size angular frequency
w = np.arange(N_POINTS / 2) * dW  # Angular frequency
