"""All functions / methods to calculate the IS spectra.

Generated using SMOP  0.41.
"""

import numpy as np
import scipy.integrate as spint
import scipy
# import quadpy

import config as cf


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))

    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = spint.quad(real_func, a, b, **kwargs)
    imag_integral = spint.quad(imag_func, a, b, **kwargs)
    # , real_integral[1:], imag_integral[1:])
    return real_integral[0] + 1j*imag_integral[0]


def dne(w, k0, w_c, ne, Te, ny_e, Mi, Ti, ny_i, B, theta):
    Fe = isspec_Fe(w, k0, w_c, ny_e, Te, theta)
    Fi = isspec_Fi(w, k0, w_c, ny_i, Ti, theta, Mi)

    w_c = w_e_gyro(np.linalg.norm([B], 2))
    # W_c = w_ion_gyro(np.linalg.norm([B], 2), np.dot(Mi, cf.M_P))
    w_p = w_plasma(ne)
    # l_D = L_Debye(ne, Te)
    Xp = np.sqrt(cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2))
    # Xp = sqrt(1 / (2 * l_D**2 * k0**2))

    dn_e = ne / (np.pi * w) * (np.imag(- Fe) * abs(1 + 2 * Xp**2 * Fi)**2 + 4 *
                               Xp**4 * np.imag(- Fi) * abs(Fe)**2) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return dn_e


# disable=pylint:too-many-arguments
def isspec_Fi(w, k, w_c, ny_i, Ti, theta, Mi):
    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...
    M_i = Mi * (cf.M_P + cf.M_N) / 2

    X = np.sqrt(M_i * w**2 / (2 * cf.K_B * Ti * k**2))
    Xi = np.sqrt(M_i * w_c**2 / (2 * cf.K_B * Ti * k**2))

    Lambda_i = ny_i / w_c

    if theta != 0:
        def Fi_integrand(y):
            return np.exp(- 1j * X / Xi * y -
                          Lambda_i * y -
                          (1 / (2 * Xi**2)) * (np.sin(theta)**2 * (1 - np.cos(y)) +
                                               1 / 2 * np.cos(theta)**2 * y**2))
        Fi = complex_quadrature(Fi_integrand, 0, np.inf, epsabs=1e-16)
    else:
        # Analytical solution to the integral
        #             /     2             2 \
        #             |    a    a b i    b  | /    /    a - b i    \       \
        # sqrt(pi) exp| - --- + ----- + --- | | erf| ------------- | i + i |
        #             \   2 c     c     2 c / |    |       /   c \ |       |
        #                                     |    | 2 sqrt| - - | |       |
        #                                     \    \       \   2 / /       /
        # ------------------------------------------------------------------
        #                                  /   c \
        #                            2 sqrt| - - |
        #                                  \   2 /
        # Where
        # a = w/w_c
        # b = ny_i/w_c (Really? looks odd to me??? BG 20161229)
        # c = cf.K_B*T*k^2/(m*w_c)
        a = w / w_c
        b = ny_i / w_c
        c = cf.K_B * Ti * k**2 / (M_i * w_c)
        Fi = np.sqrt(2 * np.pi) * np.exp(- a**2 / (2 * c) + 1j * a * b / c + b**2 / (2 * c)) * \
            ((scipy.special.erf((- a + 1j * b) / (2 * np.sqrt(- c / 2)))
              * 1j) + 1j) / (2 * np.sqrt(- c / 2))
        # Fi = np.sqrt(2 * np.pi) * np.exp(- a**2 / (2 * c) + 1j * a * b / c + b**2 / (2 * c)) *
        #                  ((Faddeeva_erf((- a + 1j * b) / (2 * np.sqrt(- c / 2))) * 1j) + 1j) / (2 * np.sqrt(- c / 2))

    Fi = 1 - (1j * X / Xi + Lambda_i) * Fi

    return Fi


def isspec_Fe(w=None, k=None, w_c=None, ny_e=None, Te=None, theta=None):
    X = np.sqrt(cf.M_E * w**2 /
                (2 * cf.K_B * Te * k**2))
    Xe = np.sqrt(cf.M_E * w_c**2 /
                 (2 * cf.K_B * Te * k**2))
    Lambda_e = ny_e / w_c

    if theta != 0:
        def Fe_integrand(y):
            return np.exp(- 1j * (X / Xe) * y - Lambda_e * y - (1 / (2 * Xe**2))
                          * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
        Fe = complex_quadrature(Fe_integrand, 0, np.inf, epsabs=1e-16)
    else:
        # Analytical solution to the integral
        #             /     2             2 \
        #             |    a    a b i    b  |  /   /    a - b i    \       \
        # sqrt(pi) exp| - --- + ----- + --- | | erf| ------------- | i + i |
        #             \   2 c     c     2 c / |    |       /   c \ |       |
        #                                     |    | 2 sqrt| - - | |       |
        #                                     \    \       \   2 / /      /
        # ------------------------------------------------------------------
        #                                  /   c \
        #                            2 sqrt| - - |
        #                                  \   2 /
        # Where
        # a = w/w_c
        # b = ny_e/w_c (Really? looks odd to me??? BG 20161229)
        # c = cf.K_B*T*k^2/(m*w_c)
        a = w / w_c
        b = ny_e / w_c
        c = np.dot(np.dot(cf.K_B, Te), k**2) / (np.dot(cf.M_E, w_c))
        Fe = np.sqrt(2 * np.pi) * np.exp(- a**2 / (2 * c) + 1j * a * b / c + b**2 / (2 * c)) * \
            ((scipy.special.erf((- a + 1j * b) / (2 * np.sqrt(- c / 2)))
              * 1j) + 1j) / (2 * np.sqrt(- c / 2))

    Fe = 1 - (1j * X / Xe + Lambda_e) * Fe

    return Fe


def isspec_ne(f=None, f0=None, Ne=None, Te=None, Nu_e=None, mi=None, Ti=None, Nu_i=None, B=None, theta=None):
    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), (mi * cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2)
    Xp = np.sqrt(1 / (2 * l_D**2 * k0**2))

    Is = np.copy(w)
    for i_w in np.arange(1, len(w)).reshape(-1):
        Fe = isspec_Fe(w[i_w], k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w[i_w], k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = Ne / np.pi / w[i_w] * (np.imag(- Fe) * abs(1 + (2 * Xp**2 * Fi))**2 + (
            (4 * Xp**4 * np.imag(- Fi)) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return Is


def isspec_ro(f=None, f0=None, Ne=None, Te=None, Nu_e=None, mi=None, Ti=None, Nu_i=None, B=None, theta=None):
    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), (mi * cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2)
    Xp = np.sqrt(1 / (2 * l_D**2 * k0**2))

    Is = np.copy(w)
    for i_w in np.arange(1, len(w)).reshape(-1):
        Fe = isspec_Fe(w(i_w), k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w(i_w), k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = (cf.Q_E**2 * Ne) / np.pi / w[i_w] * (np.imag(- Fe) +
                                                       np.imag(- Fi)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return Is


def L_Debye(*args):
    nargin = len(args)
    if nargin == 1:
        n_e = args[0]
    elif nargin == 2:
        n_e = args[0]
        T_e = args[1]
    elif nargin == 3:
        n_e = args[0]
        T_e = args[1]
        T_i = args[2]

    Ep0 = 1e-09 / 36 / np.pi

    if nargin < 3:
        LD = np.sqrt(Ep0 * cf.K_B * T_e /
                     (max(0, n_e) * cf.Q_E**2))
    else:
        LD = np.sqrt(Ep0 * cf.K_B /
                     ((max(0, n_e) / T_e + max(0, n_e) / T_i) / cf.Q_E**2))

    return LD


def y_integrand(y=None, w=None, k=None, w_g=None, ny_coll=None, kBT=None, m=None, theta=None):
    f = np.exp((- 1j * w / w_g * y) - ny_coll / w_g * y - kBT * k / (m * w_g**2)
               * (np.sin(theta)**2 * (1 - np.cos(y)) + y**2 / 2 * np.cos(theta)))

    return f


def w_plasma(n_e=None):
    w_e = np.sqrt(max(0, n_e) * cf.Q_E**2 / cf.M_E)

    return w_e


def w_ion_gyro(B=None, m_ion=None):
    w_e = cf.Q_E * B / m_ion

    return w_e


def w_e_gyro(B=None):
    w_e = cf.Q_E * B / cf.M_E

    return w_e
