"""All functions / methods to calculate the IS spectra.
"""

import numpy as np


@staticmethod
def dne(w=None, k0=None, w_c=None, ne=None, Te=None, ny_e=None, Mi=None, Ti=None, ny_i=None, B=None, theta=None, *args, **kwargs):
    # varargin = dne.varargin
    # nargin = dne.nargin

    c0 = 299792458  # Speed of light
    m_e = 9.10938291e-31  # electron rest mass [kg]
    m_p = 1.672621778e-27  # proton rest mass [kg]
    m_n = 1.674927352e-27  # neutron rest mass [kg]
    q_e = 1.602176565e-19  # elementary charge [C]

    kB = 1.380662e-23  # Boltzmann constant [J/K]
    my_0 = 4 * np.pi * 1e-7  # Permeability [Vs/Am]
    Ep_0 = 1 / (my_0*c0 ^ 2)  # Permittivity [As/Vm]

    Fe = isspec_Fe(w, k0, w_c, ny_e, Te, theta)
    Fi = isspec_Fi(w, k0, w_c, ny_i, Ti, theta, Mi)

    w_c = w_e_gyro(norm(B))
    W_c = w_ion_gyro(norm(B), np.dot(Mi, m_p))
    w_p = w_plasma(ne)
    l_D = L_Debye(ne, Te)
    Xp = (np.dot(m_e, w_p ** 2) /
          (np.dot(np.dot(np.dot(2, kB), Te), k0 ** 2)))**(1 / 2)
    # Xp = sqrt(1/(2*l_D^2*k0^2));

    dn_e = np.multiply(ne / (np.dot(pi, w)), (np.multiply(imag(- Fe), abs(1 + np.dot(np.dot(2, Xp ** 2.0), Fi)) ** 2) +
                                              np.multiply(np.dot(np.dot(4, Xp ** 4), imag(- Fi)), abs(Fe) ** 2))) / abs(1 + np.dot(np.dot(2, Xp ** 2), (Fe + Fi))) ** 2

    return dn_e


@function
def isspec_Fi(w=None, k=None, w_c=None, ny_i=None, Ti=None, theta=None, Mi=None, *args, **kwargs):
    # varargin = isspec_Fi.varargin
    # nargin = isspec_Fi.nargin

    kB = 1.380662e-23  # Boltzmann constant [J/K]
    # my_0 = 4 * np.pi * 1e-7  # Permeability [Vs/Am]
    # Ep_0 = 1 / (my_0 * c0**2)  # Permittivity [As/Vm]
    # q_e = 1.602176565e-19  # Elementary charge [C]
    m_e = 9.10938291e-31  # electron rest mass [kg]
    m_p = 1.672621778e-27  # proton rest mass [kg]
    m_n = 1.674927352e-27  # neutron rest mass [kg]

    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...
    M_i = np.dot(Mi, (m_p + m_n)) / 2

    X = np.sqrt(np.dot(M_i, w ** 2) /
                (np.dot(np.dot(np.dot(2, kB), Ti), k ** 2)))
    Xi = np.sqrt(np.dot(M_i, w_c ** 2) /
                 (np.dot(np.dot(np.dot(2, kB), Ti), k ** 2)))

    Lambda_i = ny_i / w_c

    if theta != 0:
        def Fi_integrand(y=None):
            return np.exp(np.dot(np.dot(- 1j, (X / Xi)), y) - np.dot(Lambda_i, y) - np.dot(
            (1 / (np.dot(2, Xi ** 2))), (np.dot(sin(theta) ** 2, (1 - np.cos(y))) + np.dot(np.dot(1 / 2, np.cos(theta) ** 2), y ** 2))))

        Fi = quadgk(Fi_integrand, 0, inf, 'AbsTol', 1e-16)
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
        # c = kB*T*k^2/(m*w_c)
        a = w / w_c
        b = ny_i / w_c
        c = np.dot(np.dot(kB, Ti), k**2) / (np.dot(M_i, w_c))
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


def isspec_ne(f=None,f0=None,Ne=None,Te=None,Nu_e=None,mi=None,Ti=None,Nu_i=None,B=None,theta=None,*args,**kwargs):
    # varargin = isspec_ne.varargin
    # nargin = isspec_ne.nargin

    c0 = 299792458  # Speed of light [m/s]
    m_e     = 9.10938291e-31  # electron rest mass [kg]
    m_p     = 1.672621778e-27  # proton rest mass [kg]
    m_n     = 1.674927352e-27  # neutron rest mass [kg]
    q_e     = 1.602176565e-19  # elementary charge [C]

    kB	= 1.380662e-23  # Boltzmann constant [J/K]
    my_0    = 4 * np.pi * 1e-7  # Permeability [Vs/Am]
    Ep_0    = 1 / (my_0 * c0**2)  # Permittivity [As/Vm]

    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    M_i = mi * (m_p + m_n) / 2
    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...

    k0 = w0 / c0
    w_c = w_e_gyro(norm(B))
    W_c = w_ion_gyro(norm(B),np.dot(mi,m_p))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne,Te)
    Xp = (np.dot(m_e,w_p ** 2) / (np.dot(np.dot(np.dot(2,kB),Te),k0**2)))
    Xp = np.sqrt(1 / (np.dot(np.dot(2, l_D**2), k0**2)))
    for i_w in np.arange(1,length(w)).reshape(-1):
        Fe = isspec_Fe(w(i_w), k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w(i_w), k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = np.dot(Ne / pi / w(i_w), (np.dot(np.imag(- Fe), abs(1 + np.dot(np.dot(2, Xp**2), Fi))**2) + np.dot(np.dot(np.dot(4, Xp**4), np.imag(- Fi)), abs(Fe)**2))) / abs(1 + np.dot(np.dot(2, Xp**2),(Fe + Fi)))**2

    return Is


def isspec_ro(f=None,f0=None,Ne=None,Te=None,Nu_e=None,mi=None,Ti=None,Nu_i=None,B=None,theta=None,*args,**kwargs):
    # varargin = isspec_ro.varargin
    # nargin = isspec_ro.nargin

    c0 = 299792458
    m_e = 9.10938291e-31
    m_p = 1.672621778e-27
    m_n = 1.674927352e-27
    q_e = 1.602176565e-19
    kB = 1.380662e-23

    my_0 = np.dot(np.dot(4, np.pi),1e-07)

    Ep_0 = 1 / (np.dot(my_0, c0**2))

    w = np.dot(np.dot(f, 2), np.pi)
    w0 = np.dot(np.dot(f0, 2), np.pi)
    M_i = np.dot(mi, (m_p + m_n)) / 2

    k0 = w0 / c0
    w_c = w_e_gyro(norm(B))
    W_c = w_ion_gyro(norm(B), np.dot(mi, m_p))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = (np.dot(m_e, w_p**2) / (np.dot(np.dot(np.dot(2, kB), Te), k0**2)))
    Xp = sqrt(1 / (np.dot(np.dot(2, l_D**2), k0**2)))
    for i_w in np.arange(1, np.length(w)).reshape(-1):
        Fe = isspec_Fe(w(i_w), k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w(i_w), k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = np.dot(np.dot(q_e ** 2,Ne) / pi / w(i_w),(imag(- Fe) + imag(- Fi))) / abs(1 + np.dot(np.dot(2,Xp ** 2),(Fe + Fi))) ** 2

    return Is