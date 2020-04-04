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


cdef np.ndarray[np.double_t, mode="c", ndim=1] f_0_gauss_shell(np.ndarray[np.double_t, ndim=1, mode="c"] v, dict params):
    cdef int T = params['T']
    cdef double m = params['m']
    cdef double k = const.k
    cdef double vth = np.sqrt(T * k / m)
    cdef double A = (2 * np.pi * vth**2)**(- 3 / 2) / 2
    cdef np.ndarray[np.double_t, ndim=1] func = A * np.exp(- (np.sqrt(v**2) - 5 * vth)**2 / (2 * vth**2)) + 10 * f_0_maxwell(v, params)
    return func / 11


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef np.ndarray[np.double_t, mode="c", ndim=1] v_int(np.ndarray[np.double_t, mode="c", ndim=1] y, dict params):
    cdef Py_ssize_t y_max = y.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] res
    res = np.zeros(y_max, dtype=np.double)
    # cdef double[:] res_view = res
    cdef int V_MAX = 1000000
    cdef int o = cf.ORDER
    cdef Py_ssize_t v_size = 10000
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] v
    v = np.linspace(0, V_MAX**(1 / o), v_size, dtype=np.double)**o
    cdef np.ndarray[np.double_t, ndim=1] f = f_0_gauss_shell(v, params)
    cdef Py_ssize_t i
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] s
    # s = np.zeros(v_size, dtype=np.double)
    for i in range(y_max):
        s = v * np.sin(p(y[i], params) * v) * f
        res[i] = si.simps(s, v)
    return res


cdef double p(double j, dict params):
    cdef double k_perp = cf.K_RADAR * np.sin(cf.I_P['THETA'])
    cdef double k_par = cf.K_RADAR * np.cos(cf.I_P['THETA'])
    cdef double w_c = params['w_c']
    return np.sqrt(2 * k_perp**2 / w_c**2 * (1 - np.cos(j * w_c)) + k_par**2 * j**2)


cdef np.ndarray[np.double_t, mode="c", ndim=1] p_d(np.ndarray[np.double_t, mode="c", ndim=1] y, dict params):
    # At y=0 we get 0/0, but in the limit as y tends to zero, we get frac = |k| * |w_c| / np.sqrt(w_c**2) (from above, opposite sign from below)
    cdef double cos_t = np.cos(cf.I_P['THETA'])
    cdef double sin_t = np.sin(cf.I_P['THETA'])
    cdef double w_c = params['w_c']
    cdef double k_r = abs(cf.K_RADAR)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] num
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] den
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] out
    num = k_r * abs(w_c) * (cos_t**2 * w_c * y + sin_t**2 * np.sin(w_c * y))
    den = w_c * (cos_t**2 * w_c**2 * y**2 - 2 * sin_t**2 * np.cos(w_c * y) + 2 * sin_t**2)**.5
    cdef double first = np.sign(y[-1]) * k_r * abs(w_c) / np.sqrt(w_c**2)
    cdef Py_ssize_t count_hack = 0
    while 1:
        if den[count_hack] == 0:
            den[count_hack] = first
            count_hack += 1
        else:
            break
    # out = np.array([])
    # np.seterr(divide='warn')
    # warnings.filterwarnings('error')
    # while 1:
    #     try:
    #         num[count_hack:] / den[count_hack:]
    #     except RuntimeWarning:  # ZeroDivisionError:
    #         count_hack += 1
    #         out = np.r_[out, first]
    #     else:
    #         break
    # second = num[count_hack:] / den[count_hack:]
    # out = np.r_[out, second]
    out = num / den
    return out


# params = {'nu': Lambda_s * w_c, 'm': m, 'T': T, 'w_c': w_c, 'kappa': kappa}
def long_calc(np.ndarray[np.double_t, mode="c", ndim=1] y, dict params):
    """Based on eq. (12) of Mace (2003).

    Arguments:
        y {np.ndarray} -- integration variable, 1D array of floats
        params {dict} -- dict of all plasma parameters

    Returns:
        np.ndarray -- the value of the integrand going into the integral in eq. (12)
    """
    return p_d(y, params) * v_int(y, params)
    # return p_d(y, params) * int_cy.v_int(y, params)
