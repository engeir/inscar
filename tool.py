import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import scipy.integrate as si
import scipy.constants as const

import config as cf
import parallelization as para
import integrand_functions as intf


def simpson(integrand, w, w_c, m, T, Lambda_s, T_MAX, kappa):
    t = np.linspace(0, T_MAX**(1 / cf.ORDER), int(cf.N_POINTS))**cf.ORDER
    params = {'nu': Lambda_s * w_c, 'm': m, 'T': T, 'w_c': w_c, 'kappa': kappa}
    f = integrand(t, params)
    val = np.exp(- 1j * w * t) * f
    # print(w, round(w) % 100)
    # if round(w) % 1000 < 100:
    #     plt.figure()
    #     plt.plot(t, val)
    #     plt.show()

    sint = si.simps(val, t)
    return sint


# DEPRECATED FUNCTION (see parallelization.py -- integrate)
def integrate(w_c, m, T, Lambda_s, T_MAX, function):
    res = np.zeros(len(cf.w), dtype=np.complex128)
    for c, v in enumerate(cf.w):
        res[c] = simpson(function, v, w_c, m, T, Lambda_s, T_MAX)
    F = 1 - (1j * cf.w + Lambda_s * w_c) * res
    return F


# DEPRECATED FUNCTION (see parallelization.py -- integrate)
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


def make_F(dt_s, w_c, Lambda_s, MT, function=intf.F_s_integrand):
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
    t = np.arange(cf.N_POINTS) * dt_s
    if function == intf.F_s_integrand:
        params = {'nu': Lambda_s * w_c, 'm': MT[0], 'T': MT[1], 'w_c': w_c}
        F = function(t, params)
        F = chirpz(F, cf.N_POINTS, dt_s, 0, 1)
        F = 1 - (1j * cf.w + Lambda_s * w_c) * F
    elif function == intf.kappa_gordeyev:
        params = {'w_c': w_c, 'm': MT[0], 'T': MT[1]}
        F = function(t, params)
        F = chirpz(F, cf.N_POINTS, dt_s, 0, 1)
        F /= (2**(cf.KAPPA - 1 / 2) * sps.gamma(cf.KAPPA + 1 / 2))
        F = 1 - (1j * cf.w + Lambda_s * w_c) * F
    elif function == intf.maxwell_gordeyev:
        params = {'w_c': w_c, 'm': MT[0], 'T': MT[1], 'nu': Lambda_s * w_c}
        F = function(t, params)
        F = chirpz(F, cf.N_POINTS, dt_s, 0, 1)
        F = 1 - (1j * cf.w + Lambda_s * w_c) * F
    return F


def isr_spectrum(version, kappa=5):
    """Calculate a ISR spectrum using the theory presented by Hagfors [1961].

    Arguments:
        version {str} -- decide which integral to use when calculating ISR spectrum

    Raises:
        SystemError: if the version is not valid / not found among the existing versions, an error is raised

    Returns:
        1D array -- two one dimensional numpy arrays for the frequency domain and the values of the spectrum
    """
    versions = ['hagfors', 'kappa', 'maxwell']
    try:
        if not version in versions:
            raise SystemError
        else:
            print(f'Using version "{version}"', flush=True)
    except Exception:
        version_error(version, versions)
    if version == 'hagfors':
        func = intf.F_s_integrand
    elif version == 'kappa':
        func = intf.kappa_gordeyev
    elif version == 'maxwell':
        func = intf.maxwell_gordeyev
    w_c = w_e_gyro(np.linalg.norm([cf.I_P['B']], 2))
    M_i = cf.I_P['MI'] * (const.m_p + const.m_n) / 2
    W_c = w_ion_gyro(np.linalg.norm([cf.I_P['B']], 2), M_i)
    Lambda_e, Lambda_i = cf.I_P['NU_E'] / w_c, cf.I_P['NU_I'] / W_c
    # dt_e = cf.T_MAX_e / cf.N_POINTS
    # dt_i = cf.T_MAX_i / cf.N_POINTS

    # Chirp-z transform
    # Fe = make_F(dt_e, w_c, Lambda_e, [const.m_e, cf.I_P['T_E']], function=func)
    # Fi = make_F(dt_i, W_c, Lambda_i, [M_i, cf.I_P['T_I']], function=func)
    # Time comparison between linear and parallel implementation
    # fe_params = {'w_c': w_c, 'lambda': Lambda_e, 'function': func}
    # fi_params = {'w_c': W_c, 'm': M_i, 'lambda': Lambda_i, 'function': func}
    # Fe, Fi = compare_linear_parallel(fe_params, fi_params)
    # Simpson integration in linear
    # Fe = integrate(w_c, const.m_e, cf.I_P['T_E'], Lambda_e, cf.T_MAX_e, function=func)
    # Fi = integrate(W_c, M_i, cf.I_P['T_I'], Lambda_i, cf.T_MAX_i, function=func)
    # Simpson integration in parallel
    Fe = para.integrate(
        w_c, const.m_e, cf.I_P['T_E'], Lambda_e, cf.T_MAX_e, function=func, kappa=kappa)
    Fi = para.integrate(
        W_c, M_i, cf.I_P['T_I'], Lambda_i, cf.T_MAX_i, function=func, kappa=kappa)
    # params_e = {'nu': cf.I_P['NU_E'], 'm': const.m_e, 'T': cf.I_P['T_E'], 'w_c': w_c}
    # params_i = {'nu': cf.I_P['NU_I'], 'm': M_i, 'T': cf.I_P['T_I'], 'w_c': W_c}
    # Fe = intf.two_p_isotropic_kappa(params_e)
    # Fi = intf.two_p_isotropic_kappa(params_i)

    Xp = np.sqrt(
        1 / (2 * L_Debye(cf.I_P['NE'], cf.I_P['T_E'])**2 * cf.K_RADAR**2))
    f_scaled = cf.f / 1e6
    Is = cf.I_P['NE'] / (np.pi * cf.w) * (np.imag(- Fe) * abs(1 + 2 * Xp**2 * Fi)**2 + (
        4 * Xp**4 * np.imag(- Fi) * abs(Fe)**2)) / abs(1 + 2 * Xp**2 * (Fe + Fi))**2

    return f_scaled, abs(Is)


def H_func(X, kappa, X_p, F_e, F_i):
    """Calculate H(x) from eq. (49) in Hagfors' paper.

    Arguments:
        X {np.ndarray} -- the variables along the first axis
        kappa {int} -- square root of the ratio of ion to electron mass
        X_p {float} -- plasma frequency times other constants
        F_e {np.ndarray} -- F function for electrons
        F_i {np.ndarray} -- F function for ions

    Returns:
        np.ndarray -- a function for H(x)
    """
    num = np.exp(- X**2) * abs(1 + 2 * X_p**2 * F_i)**2 + 4 * \
        X_p**2 * kappa * np.exp(- kappa**2 * X**2) * abs(F_e)**2
    den = abs(1 + 2 * X_p**2 * (F_e + F_i))**2
    return num / den


def H_spectrum(version, test=False, k=5):
    """Make plots similar to fig. (2) in Hagfors' paper.

    Arguments:
        version {str} -- decide which integral to use when calculating ISR spectrum

    Raises:
        SystemError: if the version is not valid / not found among the existing versions, an error is raised
    """
    versions = ['hagfors', 'kappa', 'maxwell']
    try:
        if not version in versions:
            raise SystemError
        else:
            print(f'Using version "{version}"', end='    \r')
    except Exception:
        version_error(version, versions)
    # NOTE: if Hagfors is used, sub in 1e-2 for 1e2 (he scales with w_c).
    # W_c approx 2.0954e+02
    # w_c approx 6.1559e+06
    if version == 'hagfors':
        func = intf.F_s_integrand
    elif version == 'kappa':
        func = intf.kappa_gordeyev
    elif version == 'maxwell':
        func = intf.maxwell_gordeyev
    w_c = w_e_gyro(np.linalg.norm([cf.I_P['B']], 2))
    W_c = w_ion_gyro(np.linalg.norm(
        [cf.I_P['B']], 2), (cf.I_P['MI'] * const.m_p))
    M_i = cf.I_P['MI'] * (const.m_p + const.m_n) / 2
    Lambda_e, Lambda_i = 0, 0

    Fe = para.integrate(w_c, const.m_e, cf.I_P['T_E'], Lambda_e,
                        cf.T_MAX_e, function=func, kappa=k)
    Fi = para.integrate(
        W_c, M_i, cf.I_P['T_I'], Lambda_i, cf.T_MAX_i, function=func, kappa=k)
    _, X = make_X(w_c, const.m_e, cf.I_P['T_E'])
    X, F = clip(X, 1e-4, 1e1, Fe, Fi)
    Fe, Fi = F[0], F[1]
    if test:
        H = H_func(X, 43, 300, Fe, Fi)
        return X, H

    kappa = [43, 172]
    leg = []
    plt.figure(figsize=(14, 8))
    for c, k in enumerate(kappa):
        plt.subplot(1, 2, c + 1)
        for X_p in [300, 3., 1., .5, .1, .03]:
            H = H_func(X, k, X_p, Fe, Fi)
            plt.loglog(X, H)
            if k == 43:
                leg.append(f'X_p = {X_p}')
        plt.ylim([1e-3, 1e2])
        plt.legend(leg, loc='lower left')
        plt.title(f'Kappa = {k}')
        plt.xlabel('f')
        plt.ylabel('H(f)')
        plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()


def clip(array, mini, maxi, *args):
    """Clip out only the interesting part of the frequency spectrum for the H(f) plots.

    Arguments:
        array {np.ndarray} -- the variables along the first axis
        mini {float} -- minimum value that is kept
        maxi {float} -- maximum value that is kept

    Returns:
        np.ndarray and list of arrays -- both the clipped first axis and y axis
    """
    mask = (array >= mini) & (array <= maxi)
    if args:
        outs = []
        for func in args:
            outs.append(func[mask])
        array = array[mask]
        return array, outs

    array = array[mask]
    return array


def make_X(w_c, M, T):
    """Calculate the X_s and X functions.

    Arguments:
        w_c {float} -- gyro frequency
        M {float} -- mass
        T {float} -- temperature

    Returns:
        float, 1D array -- the value of X_s and the X function as a function of frequency
    """
    X_s = np.sqrt(M * w_c**2 / (2 * const.k * T * cf.K_RADAR**2))
    X = np.sqrt(M * cf.w**2 / (2 * const.k * T * cf.K_RADAR**2))

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
        LD = np.sqrt(Ep0 * const.k * T_e /
                     (max(0, n_e) * const.e**2))
    else:
        LD = np.sqrt(Ep0 * const.k /
                     ((max(0, n_e) / T_e + max(0, n_e) / T_i) / const.e**2))

    return LD


def w_ion_gyro(B, m_ion):
    """Ion gyro frequency as a function of magnetic field strength and ion mass.

    Arguments:
        B {float} -- magnetic field strength
        m_ion {float} -- ion mass

    Returns:
        float -- ion gyro frequency
    """
    w_e = const.e * B / m_ion

    return w_e


def w_e_gyro(B):
    """Electron gyro frequency as a function of magnetic field strength.

    Arguments:
        B {float} -- magnetic field strength

    Returns:
        float -- electron gyro frequency
    """
    w_e = const.e * B / const.m_e

    return w_e


def version_error(version, versions):
    exc_type, _, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f'{exc_type} error in file {fname}, line {exc_tb.tb_lineno}')
    print(f'The version is wrong: {version} not found in {versions}')
    exit()


def compare_linear_parallel(fe_params, fi_params):
    tt = time.localtime()
    t0 = tt[3] * 3600 + tt[4] * 60 + tt[5]
    fe = integrate(fe_params['w_c'], const.m_e, cf.I_P['T_E'],
                   fe_params['lambda'], cf.T_MAX_e, function=fe_params['function'])
    tt = time.localtime()
    t1 = tt[3] * 3600 + tt[4] * 60 + tt[5]
    print('')
    print('Linear, Fe: ', t1 - t0, ' [s]')
    fi = integrate(fi_params['w_c'], fi_params['m'], cf.I_P['T_I'],
                   fi_params['lambda'], cf.T_MAX_i, function=fi_params['function'])
    tt = time.localtime()
    t2 = tt[3] * 3600 + tt[4] * 60 + tt[5]
    print('Linear, Fi: ', t2 - t1, ' [s]')
    Fe = para.integrate(fe_params['w_c'], const.m_e, cf.I_P['T_E'], fe_params['lambda'],
                        cf.T_MAX_e, function=fe_params['function'])
    tt = time.localtime()
    t3 = tt[3] * 3600 + tt[4] * 60 + tt[5]
    print('Parallel, Fe: ', t3 - t2, ' [s]')
    Fi = para.integrate(fi_params['w_c'], fi_params['m'], cf.I_P['T_I'],
                        fi_params['lambda'], cf.T_MAX_i, function=fi_params['function'])
    tt = time.localtime()
    t4 = tt[3] * 3600 + tt[4] * 60 + tt[5]
    print('Parallel, Fi: ', t4 - t3, ' [s]')
    print('Claim: "All elements from linear is equal to the equivalent elements from parallel."\n * Fe:',
          all(Fe == fe), '\n * Fi:', all(Fi == fi))
    return Fe, Fi
