import numpy as np

import config as cf
import matplotlib.pyplot as plt


def chirpz(g, n, dt, wo, w_c):
    """transforms g(t) into G(w)
    g(t) is n-point array and output G(w) is (n/2)-points starting at wo
    dt and dw, sampling intervals of g(t) and G(w), and wo are
    prescribed externally in an independent manner
    --- see Li, Franke, Liu [1991]

    Function written by Erhan Kudeki.

    Eirik Enger 23_01_2020:
    Edited to accept a value for alpha (Li, Franke, Liu [1991]).
    Here, alpha = 1 / w_c.

    Arguments:
    g {1D array} -- ACF ⟨e^{jkΔr}⟩ (dim: (N,))
    n {int} -- number of data points / samples along time axis
    dt {float} -- step size in time (dt = T_MAX / n)
    dw {float} -- step size in frequency (dw = 2 pi (fmax - fo) / (N / 2), where fo = 0.)
    wo {float} -- center frequency along axis (wo = 2 pi f0)
    """
    g[0] = 0.5 * g[0]  # first interval is over dt/2, and hence ...
    W = np.exp(-1j * cf.dW * dt * np.arange(n)**2 / (2. * w_c))
    S = np.exp(-1j * wo * dt * np.arange(n) / w_c)  # frequency shift by wo
    x = g * W * S
    y = np.conj(W)
    x[int(n / 2):] = 0.
    # treat 2nd half of x and y specially
    y[int(n / 2):] = y[0: int(n / 2)][::-1]
    xi = np.fft.fft(x)
    yi = np.fft.fft(y)
    G = dt * W * np.fft.ifft(xi * yi)  # in MATLAB use ifft then fft (EK)
    return G[0: int(n / 2)]


def F_s_integrand(y, X_s, Lambda_s):
    """Calculate the integral in the expression for F_i in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    W = np.exp(- Lambda_s * y - (1 / (2 * X_s**2)) * (np.sin(cf.THETA)
                                                      ** 2 * (1 - np.cos(y)) + 1 / 2 * np.cos(cf.THETA)**2 * y**2))
    return W


def make_F(dt_s, w_c, Lambda_s, MT, function=F_s_integrand):
    """Calculate the F function according to Hagfors using the chirp-z transform.

    Arguments:
        dt_s {float} -- time step size
        w_c {float} -- gyro frequency
        Lambda_s {float} -- Debye length
        MT {list} -- list of floats; first value is mass, second is temperature

    Keyword Arguments:
        function {function} -- reference to a function representing the integrand (default: {F_s_integrand})

    Returns:
        1D array -- the F function as a function of frequency
    """
    X_s, X = make_X(w_c, MT[0], MT[1])
    t = np.arange(cf.N_POINTS) * dt_s
    F = function(t, X_s, Lambda_s)
    F = chirpz(F, cf.N_POINTS, dt_s, 0, w_c)
    F = 1 - (1j * X / X_s + Lambda_s) * F

    return F


def isr_spectrum():
    """Calculate a ISR spectrum using the theory presented by Hagfors [1961].

    Returns:
        1D array -- two one dimensional numpy arrays for the frequency domain and the values of the spectrum
    """
    w_c = w_e_gyro(np.linalg.norm([cf.B], 2))
    W_c = w_ion_gyro(np.linalg.norm([cf.B], 2), (cf.MI * cf.M_P))
    Xp = np.sqrt(1 / (2 * L_Debye(cf.NE, cf.T_E)**2 * cf.K_RADAR**2))
    M_i = cf.MI * (cf.M_P + cf.M_N) / 2
    Lambda_e, Lambda_i = cf.NU_E / w_c, cf.NU_I / W_c
    dt_e = cf.T_MAX / cf.N_POINTS
    dt_i = dt_e * 1e-2

    NN = cf.N_POINTS
    for _ in range(3):
        dW = 2 * np.pi * (6e6 - 0) / (NN / 2)
        N_min = cf.T_MAX * (NN - 1) * dW / np.pi
        print('%1.3e' % N_min)
        NN = N_min

    Fe = make_F(dt_e, w_c, Lambda_e, [cf.M_E, cf.T_E])
    Fi = make_F(dt_i, W_c, Lambda_i, [M_i, cf.T_I])

    f_scaled = cf.f / 1e6
    Is = cf.NE / np.pi / cf.w * (np.imag(- Fe) * abs(1 + (2 * Xp**2 * Fi))**2 + (
        (4 * Xp**4 * np.imag(- Fi)) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return f_scaled, abs(Is)


def H_func(X, kappa, X_p, F_e, F_i):
    num = np.exp(- X**2) * abs(1 + 2 * X_p**2 * F_i)**2 + 4 * \
        X_p**2 * kappa * np.exp(- kappa**2 * X**2) * abs(F_e)**2
    den = abs(1 + 2 * X_p**2 * (F_e + F_i))**2
    return num / den


def H_spectrum():
    w_c = w_e_gyro(np.linalg.norm([cf.B], 2))
    W_c = w_ion_gyro(np.linalg.norm([cf.B], 2), (cf.MI * cf.M_P))
    M_i = cf.MI * (cf.M_P + cf.M_N) / 2
    Lambda_e, Lambda_i = 0, 0  # cf.NU_E / w_c, cf.NU_I / W_c
    dt_e = cf.T_MAX / cf.N_POINTS
    dt_i = dt_e * 1e-2

    Fe = make_F(dt_e, w_c, Lambda_e, [cf.M_E, cf.T_E])
    Fi = make_F(dt_i, W_c, Lambda_i, [M_i, cf.T_I])
    _, X = make_X(w_c, cf.M_E, cf.T_E)

    kappa = [43, 172]
    leg = []
    plt.figure()
    for c, k in enumerate(kappa):
        plt.subplot(1, 2, c + 1)
        for X_p in [300, 3., 1., .5, .1, .03]:
            H = H_func(X, k, X_p, Fe, Fi)
            plt.loglog(X, H)
            if k == 43:
                leg.append(f'X_p = {X_p}')
        plt.xlim([1e-4, 1e1])
        plt.ylim([1e-3, 1e2])
        plt.legend(leg, loc='lower left')
        plt.title(f'Kappa = {k}')
        plt.xlabel('f')
        plt.ylabel('H(f)')
        plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()


def make_X(w_c, M, T):
    """Calculate the X_s and X functions.

    Arguments:
        w_c {float} -- gyro frequency
        M {float} -- mass
        T {float} -- temperature

    Returns:
        float, 1D array -- the value of X_s and the X function as a function of frequency
    """
    X_s = np.sqrt(M * w_c**2 / (2 * cf.K_B * T * cf.K_RADAR**2))
    X = np.sqrt(M * cf.w**2 / (2 * cf.K_B * T * cf.K_RADAR**2))

    return X_s, X


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
