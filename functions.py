"""All functions / methods to calculate the IS spectra.

Generated using SMOP  0.41.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import config as cf


def chirpz(g, n, dt, dw, wo, w_c):
    """transforms g(t) into G(w)
    g(t) is n-point array and output G(w) is (n/2)-points starting at wo
    dt and dw, sampling intervals of g(t) and G(w), and wo are
    prescribed externally in an independent manner
    --- see Li, Franke, Liu [1991]

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


def complex_quadrature(func, a, b, **kwargs):
    """NOT USED!

    Integrate a complex function using the scipy.integrate.quad() method.

    Arguments:
        func {method} -- the method/function that is integrated
        a {float or np.inf} -- lower bound of the integral
        b {float of np.inf} -- upper bound of the integral

    Returns:
        float -- the value of the integral
    """
    def real_func(x):
        return sp.real(func(x))

    def imag_func(x):
        return sp.imag(func(x))

    real_integral = spint.quad(real_func, a, b, **kwargs)
    imag_integral = spint.quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j * imag_integral[0]


def dne(w, k0, w_c, ne, Te, ny_e, Mi, Ti, ny_i, B, theta):
    """NOT USED!
    Calculate the spectrum according to eq. (45) in Hagfors' paper.

    Arguments:
        w {float} -- frequency
        k0 {float} -- wavenumber
        w_c {float} -- another frequency
        ne {float} -- electron number density
        Te {float} -- electron temperature
        ny_e {float} -- electron collision frequency
        Mi {float} -- ion mass
        Ti {float} -- ion temperature
        ny_i {float} -- ion collision frequency
        B {float} -- magnetic field strength
        theta {float} -- pitch angle

    Returns:
        float -- electron spectrum
    """
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
    """Calculate the F_i function from Hagfors (page 1705).

    Arguments:
        w {float} -- frequency
        k {float} -- radar wavenumber
        w_c {float} -- ion gyro frequency
        ny_i {float} -- ion collision frequency
        Ti {float} -- ion temperature
        theta {float} -- pitch angle
        Mi {int} -- ion mass

    Returns:
        float -- the value of the function F_i
    """
    # For typical ionospheric ions
    # there is an equal number oh
    # protons and neutrons, sue me...
    M_i = Mi * (cf.M_P + cf.M_N) / 2

    X = np.sqrt(M_i * w**2 / (2 * cf.K_B * Ti * k**2))
    Xi = np.sqrt(M_i * w_c**2 / (2 * cf.K_B * Ti * k**2))

    Lambda_i = ny_i / w_c

    if theta != 0:
        def Fi_integrand(y):
            """Calculate the integral in the expression for F_i in Hagfors.

            Arguments:
                y {float} -- integration variable

            Returns:
                float -- the value of the integral
            """
            W = np.exp(- Lambda_i * y - (1 / (2 * Xi**2)) * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
            # W = np.exp(- 1j * X / Xi * y - Lambda_i * y - (1 / (2 * Xi**2)) * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
            return W


        # <WORKING>
        samples = cf.N_POINTS
        t_max = cf.T_MAX
        dt = t_max / samples * 1e-2
        df = (cf.F_ION_MAX - 0) / (samples / 2)
        dw = 2 * np.pi * df
        fo = 0.
        wo = 2 * np.pi * fo
        time = np.arange(samples) * dt
        Fi_acf = Fi_integrand(time)
        # plt.figure()
        # plt.title('Ions')
        # plt.plot(time * cf.N_POINTS, Fi_acf)
        # plt.show()
        Fi = chirpz(Fi_acf, samples, dt, dw, wo, w_c)
        # plt.figure()
        # plt.title('Chirpz')
        # w = wo + np.arange(samples / 2) * dw
        # plt.plot(w / 2. / np.pi / cf.N_POINTS, np.real(Fi))
        # plt.plot(w / 2. / np.pi / cf.N_POINTS, np.imag(Fi))
        # plt.show()
        # </WORKING>
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
            ((sp.special.erf((- a + 1j * b) / (2 * np.sqrt(- c / 2)))
              * 1j) + 1j) / (2 * np.sqrt(- c / 2))

    Fi = 1 - (1j * X / Xi + Lambda_i) * Fi

    return Fi


def isspec_Fe(w, k, w_c, ny_e, Te, theta):
    """Calculate the F_e function from Hagfors (page 1705).

    Arguments:
        w {float} -- frequency
        k {float} -- radar wavenumber
        w_c {float} -- electron gyro frequency
        ny_e {float} -- electron collision frequency
        Te {float} -- electron temperature
        theta {float} -- pitch angle

    Returns:
        float -- the value of the function F_e
    """
    X = np.sqrt(cf.M_E * w**2 /
                (2 * cf.K_B * Te * k**2))
    Xe = np.sqrt(cf.M_E * w_c**2 /
                 (2 * cf.K_B * Te * k**2))
    Lambda_e = ny_e / w_c

    if theta != 0:
        def Fe_integrand(y):
            """Calculate the integral in the expression for F_e in Hagfors.

            Arguments:
                y {float} -- integration variable

            Returns:
                float -- the value of the integral
            """
            W = np.exp(- Lambda_e * y - (1 / (2 * Xe**2))
                       * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
            # W = np.exp(- 1j * (X / Xe) * y - Lambda_e * y - (1 / (2 * Xe**2))
            #            * (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
            return W

        # <WORKING>
        samples = cf.N_POINTS
        t_max = cf.T_MAX
        dt = t_max / samples
        df = (cf.F_ION_MAX - 0) / (samples / 2)
        dw = 2 * np.pi * df
        fo = 0.
        wo = 2 * np.pi * fo
        time = np.arange(samples) * dt
        Fe_acf = Fe_integrand(time)
        # plt.figure()
        # plt.title('Electrons')
        # plt.plot(time * cf.N_POINTS, Fe_acf)
        # plt.show()
        Fe = chirpz(Fe_acf, samples, dt, dw, wo, w_c)
        # plt.figure()
        # plt.title('Chirpz')
        # w = wo + np.arange(samples / 2) * dw
        # plt.plot(w / 2. / np.pi / cf.N_POINTS, np.real(Fe))
        # plt.plot(w / 2. / np.pi / cf.N_POINTS, np.imag(Fe))
        # plt.show()
        # </WORKING>
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
            ((sp.special.erf((- a + 1j * b) / (2 * np.sqrt(- c / 2)))
              * 1j) + 1j) / (2 * np.sqrt(- c / 2))

    Fe = 1 - (1j * X / Xe + Lambda_e) * Fe

    return Fe


def isspec_ne(f, f0, Ne, Te, Nu_e, mi, Ti, Nu_i, B, theta):
    """Solve eq. (45) for the incoherent scatter spectrum in the paper by Hagfors.

    Arguments:
        f {np.ndarray} -- linear frequency
        f0 {float} -- radar frequency
        Ne {float} -- electron number density
        Te {float} -- electron temperature
        Nu_e {float} -- electron collision frequency
        mi {float} -- ion mass
        Ti {float} -- ion temperature
        Nu_i {float} -- ion collision frequency
        B {float} -- magnetic field strength
        theta {float} -- pitch angle

    Returns:
        np.ndarray -- full IS spectrum over the frequency domain
    """
    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), (mi * cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2)
    Xp = np.sqrt(1 / (2 * l_D**2 * k0**2))

    # New version: working as of 23_01_2020
    samples = cf.N_POINTS
    df = (cf.F_ION_MAX - 0) / (samples / 2)
    dw = 2 * np.pi * df
    w = np.arange(samples / 2) * dw
    Fe = isspec_Fe(w, k0, w_c, Nu_e, Te, theta)
    Fi = isspec_Fi(w, k0, W_c, Nu_i, Ti, theta, mi)
    Is = Ne / np.pi / w * (np.imag(- Fe) * abs(1 + (2 * Xp**2 * Fi))**2 + (
        (4 * Xp**4 * np.imag(- Fi)) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2
    # Old version
    # for i_w in np.arange(1, len(w)).reshape(-1):
    #     Fe = isspec_Fe(w[i_w], k0, w_c, Nu_e, Te, theta)
    #     Fi = isspec_Fi(w[i_w], k0, W_c, Nu_i, Ti, theta, mi)
    #     Is[i_w] = Ne / np.pi / w[i_w] * (np.imag(- Fe) * abs(1 + (2 * Xp**2 * Fi))**2 + (
    #         (4 * Xp**4 * np.imag(- Fi)) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return Is, w


def isspec_ro(f, f0, Ne, Te, Nu_e, mi, Ti, Nu_i, B, theta):
    """Calculate the charge density variations according to eq. (47) in Hagfors.

    Arguments:
        f {np.ndarray} -- linear frequency
        f0 {float} -- radar frequency
        Ne {float} -- electron number density
        Te {float} -- electron temperature
        Nu_e {float} -- electron collision frequency
        mi {float} -- ion mass
        Ti {float} -- ion temperature
        Nu_i {float} -- ion collision frequency
        B {float} -- magnetic field strength
        theta {float} -- pitch angle

    Returns:
        np.ndarray -- the spectrum of the charge density variations
    """
    w = f * 2 * np.pi
    w0 = f0 * 2 * np.pi

    k0 = w0 / cf.C_0
    w_c = w_e_gyro(np.linalg.norm([B], 2))
    W_c = w_ion_gyro(np.linalg.norm([B], 2), (mi * cf.M_P))
    w_p = w_plasma(Ne)
    l_D = L_Debye(Ne, Te)
    Xp = cf.M_E * w_p**2 / (2 * cf.K_B * Te * k0**2)
    Xp = np.sqrt(1 / (2 * l_D**2 * k0**2))

    Is = np.zeros(w.shape)
    for i_w in np.arange(1, len(w)).reshape(-1):
        Fe = isspec_Fe(w(i_w), k0, w_c, Nu_e, Te, theta)
        Fi = isspec_Fi(w(i_w), k0, W_c, Nu_i, Ti, theta, mi)
        Is[i_w] = (cf.Q_E**2 * Ne) / np.pi / w[i_w] * (np.imag(- Fe) +
                                                       np.imag(- Fi)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return Is


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


def y_integrand(y, w, k, w_g, ny_coll, kBT, m, theta):
    """NOT USED.

    Similar to integral in eq. (B3) in Hagfors, but not quite the same.

    Arguments:
        y {float} -- integration variable
        w {float} -- frequency (parameter s) from Laplace transform
        k {float} -- wavenumber
        w_g {float} -- gyro frequency
        ny_coll {float} -- collision frequency (combined with s in eq. (30))
        kBT {float} -- k_B * T
        m {float} -- ion mass
        theta {float} -- pitch angle

    Returns:
        float -- function that is integrated in eq. (B3)
    """
    f = np.exp((- 1j * w / w_g * y) - ny_coll / w_g * y - kBT * k / (m * w_g**2)
               * (np.sin(theta)**2 * (1 - np.cos(y)) + y**2 / 2 * np.cos(theta)))

    return f


def w_plasma(n_e):
    """Plasma frequency as a function of electron number density.

    Arguments:
        n_e {float} -- electron number density

    Returns:
        float -- plasma frequency
    """
    w_e = np.sqrt(max(0, n_e) * cf.Q_E**2 / cf.M_E)

    return w_e


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
