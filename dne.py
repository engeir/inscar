# Generated with SMOP  0.41
from libsmop import *
# dne.m


@function
def dne(w=None, k0=None, w_c=None, ne=None, Te=None, ny_e=None, Mi=None, Ti=None, ny_i=None, B=None, theta=None, *args, **kwargs):
    varargin = dne.varargin
    nargin = dne.nargin

    # DNE -

    c0 = 299792458
# dne.m:5

    m_e = 9.10938291e-31
# dne.m:6

    m_p = 1.672621778e-27
# dne.m:7

    m_n = 1.674927352e-27
# dne.m:8

    q_e = 1.602176565e-19
# dne.m:9

    kB = 1.380662e-23
# dne.m:11

    my_0 = dot(dot(4, pi), 1e-07)
# dne.m:12

    Ep_0 = 1 / (dot(my_0, c0 ** 2))
# dne.m:13

    Fe = isspec_Fe(w, k0, w_c, ny_e, Te, theta)
# dne.m:15
    Fi = isspec_Fi(w, k0, w_c, ny_i, Ti, theta, Mi)
# dne.m:16
    w_c = w_e_gyro(norm(B))
# dne.m:19
    W_c = w_ion_gyro(norm(B), dot(Mi, m_p))
# dne.m:20
    w_p = w_plasma(ne)
# dne.m:21
    l_D = L_Debye(ne, Te)
# dne.m:22
    Xp = (dot(m_e, w_p ** 2) / (dot(dot(dot(2, kB), Te), k0 ** 2))) ** (1 / 2)
# dne.m:23
    # Xp = sqrt(1/(2*l_D^2*k0^2));

    dn_e = multiply(ne / (dot(pi, w)), (multiply(imag(- Fe), abs(1 + dot(dot(2, Xp ** 2.0), Fi)) ** 2) +
                                        multiply(dot(dot(4, Xp ** 4), imag(- Fi)), abs(Fe) ** 2))) / abs(1 + dot(dot(2, Xp ** 2), (Fe + Fi))) ** 2
# dne.m:26
