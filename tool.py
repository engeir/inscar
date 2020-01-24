import numpy as np

import config as cf


def chirpz(g, n, dt, dw, wo, w_c):
    """transforms g(t) into G(w)
    g(t) is n-point array and output G(w) is (n/2)-points starting at wo
    dt and dw, sampling intervals of g(t) and G(w), and wo are
    prescribed externally in an independent manner
    --- see Li, Franke, Liu [1991]

    Function written by Erhan Kudeki.

    Eirik Enger 23_01_2020:
    Edited to accept a value for p (Li, Franke, Liu [1991]).
    Here, p = w_c.

    Arguments:
    g {1D array} -- ACF ⟨e^{jkΔr}⟩ (dim: (N,))
    n {int} -- number of data points / samples along time axis
    dt {float} -- step size in time (dt = T_MAX / n)
    dw {float} -- step size in frequency (dw = 2 pi (fmax - fo) / (N / 2), where fo = 0.)
    wo {float} -- center frequency along axis (wo = 2 pi f0)
    """
    g[0] = 0.5 * g[0]  # first interval is over dt/2, and hence ...
    W = np.exp(-1j * dw * dt * np.arange(n)**2 / (2. * w_c))
    S = np.exp(-1j * wo * dt * np.arange(n))  # frequency shift by wo
    x = g * W * S
    y = np.conj(W)
    x[int(n / 2):] = 0.
    # treat 2nd half of x and y specially
    y[int(n / 2):] = y[0: int(n / 2)][::-1]
    xi = np.fft.fft(x)
    yi = np.fft.fft(y)
    G = dt * W * np.fft.ifft(xi * yi)  # in MATLAB use ifft then fft (EK)
    return G[0: int(n / 2)]


def F_s_integrand(y, X_s, Lambda_s, theta):
    """Calculate the integral in the expression for F_i in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    W = np.exp(- Lambda_s * y - (1 / (2 * X_s**2)) * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
    return W


def isr_spectrum(_, f0, Ne, Te, Nu_e, mi, Ti, Nu_i, B, theta):
    w0 = f0 * 2 * np.pi
    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), (mi * cf.M_P))
    l_D = L_Debye(Ne, Te)
    Xp = np.sqrt(1 / (2 * l_D**2 * k0**2))
    df = (cf.F_ION_MAX - 0) / (cf.N_POINTS / 2)
    dw = 2 * np.pi * df
    w = np.arange(cf.N_POINTS / 2) * dw
    Lambda_e = Nu_e / w_c
    Lambda_i = Nu_i / W_c
    wo = 0.
    t_max = cf.T_MAX
    dt_e = t_max / cf.N_POINTS
    dt_i = t_max / cf.N_POINTS * 1e-2
    t_e = np.arange(cf.N_POINTS) * dt_e
    t_i = np.arange(cf.N_POINTS) * dt_i

    M_i = mi * (cf.M_P + cf.M_N) / 2
    X_e = np.sqrt(cf.M_E * w_c**2 / (2 * cf.K_B * Te * k0**2))
    X_i = np.sqrt(M_i * W_c**2 / (2 * cf.K_B * Ti * k0**2))
    Xe = np.sqrt(cf.M_E * w**2 / (2 * cf.K_B * Te * k0**2))
    Xi = np.sqrt(M_i * w**2 / (2 * cf.K_B * Ti * k0**2))

    Fe = F_s_integrand(t_e, X_e, Lambda_e, theta)
    Fi = F_s_integrand(t_i, X_i, Lambda_i, theta)
    Fe = chirpz(Fe, cf.N_POINTS, dt_e, dw, wo, w_c)
    Fi = chirpz(Fi, cf.N_POINTS, dt_i, dw, wo, W_c)
    Fe = 1 - (1j * Xe / X_e + Lambda_e) * Fe
    Fi = 1 - (1j * Xi / X_i + Lambda_i) * Fi

    Is = Ne / np.pi / w * (np.imag(- Fe) * abs(1 + (2 * Xp**2 * Fi))**2 + (
        (4 * Xp**4 * np.imag(- Fi)) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return Is, w


def L_Debye(*args):
    """Calculate the Debye length.

    Input args may be
        n_e -- electron number density
        T_e -- electron temperature
        T_i -- ion temperature

    Returns:
        float -- the Debye length
    """
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


def w_ion_gyro(B, m_ion):
    """Ion gyro frequency as a function of magnetic field strength and ion mass.

    Arguments:
        B {float} -- magnetic field strength
        m_ion {float} -- ion mass

    Returns:
        float -- ion gyro frequency
    """
    w_e = cf.Q_E * B / m_ion

    return w_e


def w_e_gyro(B):
    """Electron gyro frequency as a function of magnetic field strength.

    Arguments:
        B {float} -- magnetic field strength

    Returns:
        float -- electron gyro frequency
    """
    w_e = cf.Q_E * B / cf.M_E

    return w_e
