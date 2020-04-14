import os
import sys
import textwrap as txt

import numpy as np
import scipy.constants as const
import scipy.integrate as si

from inputs import config as cf
from utils import integrand_functions as intf
from utils import parallelization as para


def simpson(w, y):
    val = np.exp(- 1j * w * y) * cf.ff

    sint = si.simps(val, y)
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
    func = version_check(version, vdf, kappa)
    w_c = w_e_gyro(np.linalg.norm([cf.I_P['B']], 2))
    M_i = cf.I_P['MI'] * (const.m_p + const.m_n) / 2
    W_c = w_ion_gyro(np.linalg.norm([cf.I_P['B']], 2), M_i)

    # Time comparison between linear and parallel implementation
    # fe_params = {'w_c': w_c, 'lambda': Lambda_e, 'function': func}
    # fi_params = {'w_c': W_c, 'm': M_i, 'lambda': Lambda_i, 'function': func}
    # Fe, Fi = compare_linear_parallel(fe_params, fi_params)
    # Simpson integration in parallel
    params = {'nu': cf.I_P['NU_I'], 'm': M_i,
              'T': cf.I_P['T_I'], 'w_c': W_c, 'kappa': kappa, 'vdf': vdf}
    y = np.linspace(0, cf.Y_MAX_i**(1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double)**cf.ORDER
    cf.ff = func(y, params)
    Fi = para.integrate(
        M_i, cf.I_P['T_I'], cf.I_P['NU_I'], y, function=func, kappa=kappa)
    params = {'nu': cf.I_P['NU_E'], 'm': const.m_e,
              'T': cf.I_P['T_E'], 'w_c': w_c, 'kappa': kappa, 'vdf': vdf}
    y = np.linspace(0, cf.Y_MAX_e**(1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double)**cf.ORDER
    cf.ff = func(y, params)
    Fe = para.integrate(
        const.m_e, cf.I_P['T_E'], cf.I_P['NU_E'], y, function=func, kappa=kappa)
    # params_e = {'nu': cf.I_P['NU_E'], 'm': const.m_e, 'T': cf.I_P['T_E'], 'w_c': w_c}
    # params_i = {'nu': cf.I_P['NU_I'], 'm': M_i, 'T': cf.I_P['T_I'], 'w_c': W_c}
    # Fe = intf.two_p_isotropic_kappa(params_e)
    # Fi = intf.two_p_isotropic_kappa(params_i)

    Xp_e = np.sqrt(
        1 / (2 * L_Debye(cf.I_P['NE'], cf.I_P['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))
    Xp_i = np.sqrt(
        1 / (2 * L_Debye(cf.I_P['NE'], cf.I_P['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))

    f_scaled = cf.f
    with np.errstate(divide='ignore', invalid='ignore'):
        Is = cf.I_P['NE'] / (np.pi * cf.w) * (np.imag(- Fe) * abs(1 + 2 * Xp_i**2 * Fi)**2 + (
            4 * Xp_e**4 * np.imag(- Fi) * abs(Fe)**2)) / abs(1 + 2 * Xp_e**2 * Fe + 2 * Xp_i**2 * Fi)**2

    if area and cf.I_P['F_MAX'] < 1e4:
        area = si.simps(Is, cf.f)
        print('The area under the ion line is %1.6e.' % area)

    return f_scaled, abs(Is)


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


def version_check(version, vdf, kappa):
    versions = ['hagfors', 'kappa', 'maxwell', 'long_calc']
    try:
        if not version in versions:
            raise SystemError
        print(f'Using version "{version}"', flush=True)
    except SystemError:
        sys.exit(version_error(version, versions))
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
            print(f'Using VDF "{vdf}"', flush=True)
        except Exception:
            sys.exit(version_error(vdf, vdfs, element='VDF'))
        if vdf in ['kappa', 'kappa_vol2']:
            kappa_check(kappa)
        func = intf.long_calc
    return func


def version_error(version, versions, element='version'):
    exc_type, _, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f'{exc_type} error in file {fname}, line {exc_tb.tb_lineno}')
    print(f'The {element} is wrong: "{version}" not found in {versions}')


def kappa_check(kappa):
    try:
        if kappa is None:
            raise SystemError
    except SystemError:
        print('You forgot to send in the kappa parameter.')
        sys.exit()
    if cf.I_P['NU_E'] != 0 or cf.I_P['NU_I'] != 0:
        text = f'''\
                Warning: the kappa function is defined for a collisionless plasma.
                You are using: nu_i = {cf.I_P['NU_I']} and nu_e = {cf.I_P['NU_E']}.'''
        print(txt.fill(txt.dedent(text), width=300))
