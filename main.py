"""Main script.
"""
from functions import *

import numpy as np
import matplotlib.pyplot as plt


B = 5e-4
f0 = 500e6
f_ion = np.linspace(- 150e2, 150e2, 121)
f_ion += 1
mi = 16
ne = 2e11
Nu_e = 1000
Nu_i = 100
T_e = 2500
theta = 0.1
Ti = 1500

Is = isspec_ne(f_ion + 1, f0, ne, T_e, Nu_e, mi, Ti, Nu_i, B, theta)
plt.figure()
plt.plot(f_ion, Is)
plt.show()
