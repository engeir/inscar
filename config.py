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

UPPER_LIMIT = 1e1
N_POINTS = 1e6
T_MAX = 1e3

# === Input parameters ===
LIMIT = 1500  # Subintervals in integration method
B = 5e-4  # Magnetic field strength [T]
F0 = 500e6  # Radar frequency [Hz]
F_ION_MAX = 6e6  # F0 / F_ION_MAX > 15 000
F_ION = np.linspace(0, F_ION_MAX, 1e6)  # Ion frequency [Hz]
# F_ION += 1
MI = 16  # Ion mass in atomic mass units [u]
NE = 2e11  # Electron number density [m^(-3)]
NU_E = 1000  # Electron collision frequency [Hz]
NU_I = 100  # Ion collision frequency [Hz]
T_E = 2500  # Electron temperature [K]
T_I = 1500  # Ion temperature [K]
THETA = 0.1  # 45 * np.pi / 180  # Pitch angle
