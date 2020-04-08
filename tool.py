import os
import sys
import textwrap as txt
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.integrate as si
import scipy.special as sps

import config as cf
import integrand_functions as intf
import parallelization as para
import int_cy


def simpson(w, T_MAX):
    t = np.linspace(0, T_MAX**(1 / cf.ORDER), int(cf.N_POINTS), dtype=np.double)**cf.ORDER
    val = np.exp(- 1j * w * t) * cf.ff

    sint = si.simps(val, t)
    return sint


def isr_spectrum(version, kappa=None, area=False, vdf=None):
    """Calculate a ISR spectrum using the theory presented by Hagfors [1961].

    Arguments:
        version {str} -- decide which integral to use when calculating ISR spectrum

    Raises:
        SystemError: if the version is not valid / not found among the existing versions, an error is raised

    Returns:
        1D array -- two one dimensional numpy arrays for the frequency domain and the values of the spectrum
    """
    versions = ['hagfors', 'kappa', 'maxwell', 'long_calc']
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
        kappa_check(kappa)
        func = intf.kappa_gordeyev
    elif version == 'maxwell':
        func = intf.maxwell_gordeyev
    elif version == 'long_calc':
        vdfs = ['maxwell', 'kappa', 'kappa_vol2', 'gauss_shell']
        try:
            if not vdf in vdfs:
                raise SystemError
            else:
                print(f'Using VDF "{vdf}"', flush=True)
        except Exception:
            version_error(vdf, vdfs, type='VDF')
        if vdf in ['kappa', 'kappa_vol2']:
            kappa_check(kappa)
        func = intf.long_calc
    w_c = w_e_gyro(np.linalg.norm([cf.I_P['B']], 2))
    M_i = cf.I_P['MI'] * (const.m_p + const.m_n) / 2
    W_c = w_ion_gyro(np.linalg.norm([cf.I_P['B']], 2), M_i)
    Lambda_e, Lambda_i = cf.I_P['NU_E'] / w_c, cf.I_P['NU_I'] / W_c
    # dt_e = cf.T_MAX_e / cf.N_POINTS
    # dt_i = cf.T_MAX_i / cf.N_POINTS

    # Time comparison between linear and parallel implementation
    # fe_params = {'w_c': w_c, 'lambda': Lambda_e, 'function': func}
    # fi_params = {'w_c': W_c, 'm': M_i, 'lambda': Lambda_i, 'function': func}
    # Fe, Fi = compare_linear_parallel(fe_params, fi_params)
    # Simpson integration in parallel
    t = np.linspace(0, cf.T_MAX_i**(1 / cf.ORDER), int(cf.N_POINTS), dtype=np.double)**cf.ORDER
    params = {'nu': Lambda_i * W_c, 'm': M_i,
              'T': cf.I_P['T_I'], 'w_c': W_c, 'kappa': kappa, 'vdf': vdf}
    cf.ff = func(t, params)
    Fi = para.integrate(
        W_c, M_i, cf.I_P['T_I'], Lambda_i, cf.T_MAX_i, function=func, kappa=kappa)
    t = np.linspace(0, cf.T_MAX_e**(1 / cf.ORDER), int(cf.N_POINTS), dtype=np.double)**cf.ORDER
    params = {'nu': Lambda_e * w_c, 'm': const.m_e,
              'T': cf.I_P['T_E'], 'w_c': w_c, 'kappa': kappa, 'vdf': vdf}
    cf.ff = func(t, params)
    Fe = para.integrate(
        w_c, const.m_e, cf.I_P['T_E'], Lambda_e, cf.T_MAX_e, function=func, kappa=kappa)
    # params_e = {'nu': cf.I_P['NU_E'], 'm': const.m_e, 'T': cf.I_P['T_E'], 'w_c': w_c}
    # params_i = {'nu': cf.I_P['NU_I'], 'm': M_i, 'T': cf.I_P['T_I'], 'w_c': W_c}
    # Fe = intf.two_p_isotropic_kappa(params_e)
    # Fi = intf.two_p_isotropic_kappa(params_i)

    Xp_e = np.sqrt(
        1 / (2 * L_Debye(cf.I_P['NE'], cf.I_P['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))
    Xp_i = np.sqrt(
        1 / (2 * L_Debye(cf.I_P['NE'], cf.I_P['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))

    f_scaled = cf.f
    Is = cf.I_P['NE'] / (np.pi * cf.w) * (np.imag(- Fe) * abs(1 + 2 * Xp_i**2 * Fi)**2 + (
        4 * Xp_e**4 * np.imag(- Fi) * abs(Fe)**2)) / abs(1 + 2 * Xp_e**2 * Fe + 2 * Xp_i**2 * Fi)**2

    if area and cf.I_P['F_MAX'] < 1e4:
        area = si.simps(Is, cf.f)
        print('The area under the ion line is %1.6e.' % area)

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


def L_Debye(*args, kappa=None):
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
        if kappa is None:
            LD = np.sqrt(Ep0 * const.k * T_e /
                         (max(0, n_e) * const.e**2))
        else:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e**2)
                         ) * np.sqrt((kappa - 3 / 2) / (kappa - 1 / 2))
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


def version_error(version, versions, type='version'):
    exc_type, _, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f'{exc_type} error in file {fname}, line {exc_tb.tb_lineno}')
    print(f'The {type} is wrong: "{version}" not found in {versions}')
    sys.exit()


def kappa_check(kappa):
    if kappa is None:
        print('You forgot to send in the kappa parameter.')
        sys.exit()
    if cf.I_P['NU_E'] != 0 or cf.I_P['NU_I'] != 0:
        text = f'''\
                Warning: the kappa function is defined for a collisionless plasma.
                You are using: nu_i = {cf.I_P['NU_I']} and nu_e = {cf.I_P['NU_E']}.'''
        print(txt.fill(txt.dedent(text), width=300))
