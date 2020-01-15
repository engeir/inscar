"""Main script.
"""

import numpy as np
import matplotlib.pyplot as plt

import functions as func

B = 5e-4  # Magnetic field strength [T]
f0 = 500e6  # Radar frequency [Hz]
f_ion_max = 150e2
f_ion = np.linspace(- f_ion_max, f_ion_max, 121)  # Ion frequency [Hz]
f_ion += 1
mi = 16  # Ion mass in atomic mass units [u]
ne = 2e10  # Electron number density [m^(-3)]
Nu_e = 1000  # Electron collision frequency [Hz]
Nu_i = 100  # Ion collision frequency [Hz]
T_e = 6500  # Electron temperature [K]
T_i = 4500  # Ion temperature [K]
theta = 0.1  # Pitch angle

Is = func.isspec_ne(f_ion + 1, f0, ne, T_e, Nu_e, mi, T_i, Nu_i, B, theta)
plt.figure()
plt.plot(f_ion, abs(Is))
plt.show()
