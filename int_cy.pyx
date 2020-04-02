#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
import scipy.integrate as si
import scipy.constants as const

import config as cf


cdef np.ndarray[np.double_t, ndim=1] f_0_maxwell(np.ndarray[np.double_t, ndim=1, mode="c"] v, dict params):
    cdef int T = params['T']
    cdef double m = params['m']
    cdef double k = const.k
    cdef double vth = np.sqrt(T * k / m)
    cdef double A = (2 * np.pi * vth**2)**(- 3 / 2)
    cdef np.ndarray[np.double_t, ndim=1] func = A * np.exp(- v**2 / (2 * T * k / m))
    return func


cdef np.ndarray[np.double_t, ndim=1] f_0_gauss_shell(np.ndarray[np.double_t, ndim=1, mode="c"] v, dict params):
    cdef int T = params['T']
    cdef double m = params['m']
    cdef double k = const.k
    cdef double vth = np.sqrt(T * k / m)
    cdef double A = (2 * np.pi * vth**2)**(- 3 / 2) / 2
    cdef np.ndarray[np.double_t, ndim=1] func = A * np.exp(- (np.sqrt(v**2) - 5 * vth)**2 / (2 * vth**2)) + 10 * f_0_maxwell(v, params)
    return func / 11


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def v_int(double[:] y not None, dict params):
    cdef Py_ssize_t y_max = y.shape[0]
    res = np.zeros(y_max, dtype=np.double)
    cdef double[:] res_view = res
    cdef int V_MAX = 10000000
    cdef int o = cf.ORDER
    cdef int v_size = 10000
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] v
    v = np.linspace(0, V_MAX**(1 / o), v_size, dtype=np.double)**o
    cdef np.ndarray[np.double_t, ndim=1] f = f_0_gauss_shell(v, params)
    cdef Py_ssize_t i
    s = np.zeros(v_size, dtype=np.double)
    for i in range(y_max):
        s = np.sin(p(y[i], params) * v)
        s = v * s * f
        res_view[i] = si.simps(s, v)
    return res


cdef double p(double j, dict params):
    cdef double k_perp = cf.K_RADAR * np.sin(cf.I_P['THETA'])
    cdef double k_par = cf.K_RADAR * np.cos(cf.I_P['THETA'])
    cdef double w_c = params['w_c']
    return np.sqrt(2 * k_perp**2 / w_c**2 * (1 - np.cos(j * w_c)) + k_par**2 * j**2)
