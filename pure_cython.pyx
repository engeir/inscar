#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
import scipy.constants as const

import config as cf


cdef extern from "math.h":
    double sin(double arg)


cdef extern from "math.h":
    double cos(double arg)


cdef extern from "math.h":
    double sqrt(double arg)


cdef extern from "math.h":
    double fabs(double arg)

cdef extern from "math.h":
    double exp(double arg)


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def v_int(double[:] y not None, dict params):
    cdef int T = params['T']
    cdef double m = params['m']
    cdef double k = const.k
    cdef double k_r = cf.K_RADAR
    cdef double theta = cf.I_P['THETA']
    cdef double w_c = params['w_c']
    cdef double pi = 3.141592653589793
    cdef double vth = sqrt(T * k / m)
    cdef double A = (2 * pi * vth**2)**(- 3 / 2)

    cdef Py_ssize_t y_max = y.shape[0]
    res = np.zeros(y_max, dtype=np.double)
    cdef double[:] res_view = res
    cdef int V_MAX = 10000000
    cdef int v_size = 10000
    v = cvarray(shape=(v_size,), itemsize=sizeof(int), format="i")
    cdef int [:] v_view = v
    cdef Py_ssize_t ii
    for ii in range(v_size):
        v_view[ii] = ii * 1000
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(y_max):
        for j in range(v_size):
            res_view[i] += v[j] * sin(p(y[i], k_r, theta, w_c) * v[j]) * (A / 2 * exp(- (fabs(v[j]) - 5 * vth)**2 / (2 * vth**2)) + 10 * A * exp(- v[j]**2 / (2 * T * k / m))) / 11
    return res


@cython.cdivision(True)
cdef double p(double jj, double k_r, double theta, double w_c):
    cdef double k_perp = k_r * sin(theta)
    cdef double k_par = k_r * cos(theta)
    return sqrt(2 * k_perp**2 / w_c**2 * (1 - cos(jj * w_c)) + k_par**2 * jj**2)
