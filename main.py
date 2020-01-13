"""Main script.
"""
from isspec_Fe import *
from isspec_Fi import *
from isspec_ne import *
from isspec_ro import *
from dne import *
from L_Debye import *
from w_e_gyro import *
from w_ion_gyro import *
from w_plasma import *
from y_integrand import *

import numpy as np
import matplotlib.pyplot as plt


B = 50e-4
f_0 = 500e6
# f_ion = range(- 15000, 15000, 250)
f_ion = np.linspace(- 150e2, 150e2, 121)
f_ion += 1
mi = 16
ne = 2e11
Nu_e = 1e3
Nu_i = 1e2
T_e = 25e2
theta = 0.1
Ti = 1500

Is = isspec_ne(f_ion+1,f0,ne,Te,Nu_e,mi,Ti,Nu_i,B,theta)
plt.figure()
plt.plot(f_ion, Is)
plt.show()