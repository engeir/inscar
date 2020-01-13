"""All functions / methods to calculate the IS spectra.
"""

import numpy as np
import scipy.integrate as spint

import config as cf


def dne(w=None, k0=None, w_c=None, ne=None, Te=None, ny_e=None, Mi=None, Ti=None, ny_i=None, B=None, theta=None, *args, **kwargs):
    varargin = args
    nargin = 11 + len(varargin)
    # varargin = dne.varargin
    # nargin = dne.nargin

    Fe = isspec_Fe(w, k0, w_c, ny_e, Te, theta)
    Fi = isspec_Fi(w, k0, w_c, ny_i, Ti, theta, Mi)

    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), np.dot(Mi, cf.M_P))
    w_p = w_plasma(ne)
    l_D = L_Debye(ne, Te)
    Xp = np.sqrt(cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2))
    # Xp = sqrt(1 / (2 * l_D**2 * k0**2))

    dn_e = np.multiply(ne / (np.dot(pi, w)), (np.multiply(np.imag(- Fe), abs(1 + np.dot(np.dot(2, Xp ** 2.0), Fi)) ** 2) +
                                              np.multiply(np.dot(np.dot(4, Xp ** 4), np.imag(- Fi)), abs(Fe) ** 2))) / abs(1 + np.dot(np.dot(2, Xp ** 2), (Fe + Fi))) ** 2

    return dn_e


def isspec_Fi(w=None, k=None, w_c=None, ny_i=None, Ti=None, theta=None, Mi=None, *args, **kwargs):
    # varargin = args
    # nargin = 7 + len(varargin)
    # varargin = isspec_Fi.varargin
    # nargin = isspec_Fi.nargin

    # Ep_0 = 1 / (cf.MY_0 * cf.C_0**2)  # Permittivity [As/Vm]

    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...
    M_i = Mi * (cf.M_P + cf.M_N) / 2

    X = np.sqrt(M_i * w**2 / (2 * cf.K_B * Ti * k**2))
    Xi = np.sqrt(M_i * w_c**2 / (2 * cf.K_B * Ti * k ** 2))

    Lambda_i = ny_i / w_c

    if theta != 0:
        def Fi_integrand(y=None):
            return np.exp(np.dot(np.dot(- 1j, (X / Xi)), y) - np.dot(Lambda_i, y) - np.dot(
                (1 / (np.dot(2, Xi ** 2))), (np.dot(np.sin(theta) ** 2, (1 - np.cos(y))) + np.dot(np.dot(1 / 2, np.cos(theta) ** 2), y ** 2))))

        Fi = spint.quad(Fi_integrand, 0, np.inf, epsabs=1e-16)
        Fi = np.diff(Fi)[0]
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
        c = np.dot(np.dot(cf.K_B, Ti), k**2) / (np.dot(M_i, w_c))
        Fi = np.multiply(np.dot(sqrt(np.dot(2, pi)), exp(- a**2 / (np.dot(2, c)) + np.dot(np.dot(1j, a), b) / c + b**2 / (np.dot(2, c)))),
                         (np.dot(Faddeeva_erf((- a + np.dot(1j, b)) / (np.dot(2, (- c / 2)**(1 / 2)))), 1j) + 1j)) / (np.dot(2, sqrt(- c / 2)))

    Fi = 1 - np.dot((np.dot(1j, X) / Xi + Lambda_i), Fi)

    # $$$ if w/2/pi == -1e4
    # $$$   keyboard
    # $$$ end
    # $$$ y = (1:1000)/10;
    # $$$ subplot(2,1,1)
    # $$$ plot(y,[real(Fi_integrand(y))])
    # $$$ title(w/2/pi)
    # $$$ subplot(2,1,2)
    # $$$ plot(y,[imag(Fi_integrand(y))])
    # $$$ drawnow

    return Fi


def isspec_ne(f=None, f0=None, Ne=None, Te=None, Nu_e=None, mi=None, Ti=None, Nu_i=None, B=None, theta=None, *args, **kwargs):
    varargin = args
    nargin = 10 + len(varargin)
    # varargin = isspec_ne.varargin
    # nargin = isspec_ne.nargin

    cf.C_0 = 299792458  # Speed of light [m/s]

    Ep_0 = 1 / (cf.MY_0 * cf.C_0**2)  # Permittivity [As/Vm]

    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    M_i = mi * (cf.M_P + cf.M_N) / 2
    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), np.dot(mi, cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = (np.dot(cf.M_E, w_p ** 2) / (np.dot(np.dot(np.dot(2, cf.K_B), Te), k0**2)))
    Xp = np.sqrt(1 / (np.dot(np.dot(2, l_D**2), k0**2)))
    # Is = np.zeros((len(w), 2))
    Is = np.copy(w)
    for i_w in np.arange(1, len(w)).reshape(-1):
        Fe = isspec_Fe(w[i_w], k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w[i_w], k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = np.dot(Ne / np.pi / w[i_w], (np.dot(np.imag(- Fe), abs(1 + np.dot(np.dot(2, Xp**2), Fi))**2) + np.dot(
            np.dot(np.dot(4, Xp**4), np.imag(- Fi)), abs(Fe)**2))) / abs(1 + np.dot(np.dot(2, Xp**2), (Fe + Fi)))**2

    return Is


def isspec_ro(f=None, f0=None, Ne=None, Te=None, Nu_e=None, mi=None, Ti=None, Nu_i=None, B=None, theta=None, *args, **kwargs):
    varargin = args
    nargin = 10 + len(varargin)
    # varargin = isspec_ro.varargin
    # nargin = isspec_ro.nargin


    Ep_0 = 1 / (np.dot(cf.MY_0, cf.C_0**2))

    w = np.dot(np.dot(f, 2), np.pi)
    w0 = np.dot(np.dot(f0, 2), np.pi)
    M_i = np.dot(mi, (cf.M_P + cf.M_N)) / 2

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), np.dot(mi, cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = (np.dot(cf.M_E, w_p**2) / (np.dot(np.dot(np.dot(2, cf.K_B), Te), k0**2)))
    Xp = np.sqrt(1 / (np.dot(np.dot(2, l_D**2), k0**2)))
    for i_w in np.arange(1, len(w)).reshape(-1):
        Fe = isspec_Fe(w(i_w), k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w(i_w), k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = np.dot(np.dot(cf.Q_E**2, Ne) / np.pi / w(i_w), (np.imag(- Fe) +
                                                               np.imag(- Fi))) / abs(1 + np.dot(np.dot(2, Xp**2), (Fe + Fi)))**2

    return Is


def isspec_Fe(w=None, k=None, w_c=None, ny_e=None, Te=None, theta=None, *args, **kwargs):
    varargin = args
    nargin = 6 + len(varargin)
    # varargin = isspec_Fe.varargin
    # nargin = isspec_Fe.nargin

    X = np.sqrt(np.dot(cf.M_E, w**2) /
                (np.dot(np.dot(np.dot(2, cf.K_B), Te), k ** 2)))
    Xe = np.sqrt(np.dot(cf.M_E, w_c**2) /
                 (np.dot(np.dot(np.dot(2, cf.K_B), Te), k**2)))
    Lambda_e = ny_e / w_c

    if theta != 0:
        def Fe_integrand(y):
            return np.exp(- 1j * (X / Xe) * y - Lambda_e * y - (1 / (2 * Xe**2))
                          * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
        Fe = spint.quad(Fe_integrand, 0, np.inf, epsabs=1e-16)
        Fe = np.diff(Fe)[0]
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
        # b = ny_e/w_c (Really? looks odd to me??? BG 20161229)
        # c = cf.K_B*T*k^2/(m*w_c)
        a = w / w_c
        b = ny_e / w_c
        c = np.dot(np.dot(cf.K_B, Te), k**2) / (np.dot(cf.M_E, w_c))
        Fe = np.multiply(np.dot(np.sqrt(np.dot(2, np.pi)), np.exp(- a**2 / (np.dot(2, c)) + np.dot(np.dot(1j, a), b) / c + b**2 / (np.dot(2, c)))),
                         (np.dot(Faddeeva_erf((- a + np.dot(1j, b)) / (np.dot(2, (- c / 2)**(1 / 2)))), 1j) + 1j)) / (np.dot(2, np.sqrt(- c / 2)))

    Fe = 1 - np.multiply((np.dot(1j, X) / Xe + Lambda_e), Fe)
    # $$$ y = (1:1000)/10;
    # $$$ subplot(2,1,1)
    # $$$ plot(y,[real(Fe_integrand(y))])
    # $$$ title(w/2/pi)
    # $$$ subplot(2,1,2)
    # $$$ plot(y,[imag(Fe_integrand(y))])
    # $$$ drawnow
    return Fe


def L_Debye(*args, **kwargs):  # n_e=None,T_e=None,T_i=None,
    # varargin = args
    # nargin = 3 + len(varargin)
    nargin = len(args)
    varargin = len(kwargs)
    if nargin == 1:
        n_e = args[0]
    elif nargin == 2:
        n_e = args[0]
        T_e = args[1]
    elif nargin == 3:
        n_e = args[0]
        T_e = args[1]
        T_i = args[2]
    # varargin = L_Debye.varargin
    # nargin = L_Debye.nargin

    # Help text

    Ep0 = 1e-09 / 36 / np.pi

    if nargin < 3:
        LD = np.sqrt(np.dot(np.dot(Ep0, cf.K_B), T_e) /
                     (np.dot(max(0, n_e), cf.Q_E**2)))
    else:
        LD = np.sqrt(np.dot(Ep0, cf.K_B) /
                     ((max(0, n_e) / T_e + max(0, n_e) / T_i) / cf.Q_E**2))

    return LD


def y_integrand(y=None, w=None, k=None, w_g=None, ny_coll=None, kBT=None, m=None, theta=None, *args, **kwargs):
    varargin = args
    nargin = 8 + len(varargin)
    # varargin = y_integrand.varargin
    # nargin = y_integrand.nargin

    f = np.exp(np.dot(np.dot(- 1j, w) / w_g, y) - np.dot(ny_coll / w_g, y) - np.dot(np.dot(kBT, k) /
                                                                                    (np.dot(m, w_g**2)), (np.dot(np.sin(theta)**2, (1 - np.cos(y))) + np.dot(y**2 / 2, np.cos(theta)))))

    return f


def w_plasma(n_e=None, *args, **kwargs):
    varargin = args
    nargin = 1 + len(varargin)
    # varargin = w_plasma.varargin
    # nargin = w_plasma.nargin

    Ep0 = 1e-09 / 36 / np.pi

    w_e = np.sqrt(np.dot(max(0, n_e), cf.Q_E**2) / cf.M_E)

    return w_e


def w_ion_gyro(B=None, m_ion=None, *args, **kwargs):
    varargin = args
    nargin = 2 + len(varargin)
    # varargin = w_ion_gyro.varargin
    # nargin = w_ion_gyro.nargin


    w_e = np.dot(cf.Q_E, B) / m_ion

    return w_e


def w_e_gyro(B=None, *args, **kwargs):
    varargin = args
    nargin = 1 + len(varargin)
    # varargin = w_e_gyro.varargin
    # nargin = w_e_gyro.nargin

    w_e = np.dot(cf.Q_E, B) / cf.M_E

    return w_e
