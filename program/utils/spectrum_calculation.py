"""Script containing the calculation of the power density function and other plasma parameters.

Raises:
    SystemError: if no known version of particle distribution is given
    SystemError: if, with the long_calc version, no known VDF is given
    SystemError: if, given a kappa particle distribution, no kappa index is given
"""

import os
import sys
import textwrap as txt

import numpy as np
import scipy.constants as const
import scipy.integrate as si

from inputs import config as cf
from utils import integrand_functions as intf
from utils.parallel import parallelization as para


def isr_spectrum(version, sys_set, kappa=None, area=False, vdf=None):
    """Calculate a ISR spectrum using the theory presented by Hagfors [1961].

    Arguments:
        version {str} -- decide which integral to use when calculating ISR spectrum

    Raises:
        SystemError: if the version is not valid / not found among the existing versions, an error is raised

    Returns:
        1D array -- two one dimensional numpy arrays for the frequency domain and the values of the spectrum
    """
    sys_set, p = correct_inputs(version, sys_set, {'kappa': kappa, 'vdf': vdf})
    kappa, vdf = p['kappa'], p['vdf']
    func = version_check(version, vdf, kappa, sys_set)
    # w_c = w_e_gyro(np.linalg.norm([cf.I_P['B']], 2))
    # M_i = cf.I_P['MI'] * (const.m_p + const.m_n) / 2
    # W_c = w_ion_gyro(np.linalg.norm([cf.I_P['B']], 2), M_i)
    w_c = w_e_gyro(np.linalg.norm([sys_set['B']], 2))
    M_i = sys_set['MI'] * (const.m_p + const.m_n) / 2
    W_c = w_ion_gyro(np.linalg.norm([sys_set['B']], 2), M_i)

    # Ions
    params = {'THETA': sys_set['THETA'], 'nu': sys_set['NU_I'], 'm': M_i, 'T': sys_set['T_I'],
              'w_c': W_c, 'kappa': kappa, 'vdf': vdf}
    y = np.linspace(0, cf.Y_MAX_i**(1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double)**cf.ORDER
    f_ion = intf.INT_MAXWELL()
    f_ion.initialize(y, params)
    Fi = para.integrate(M_i, sys_set['T_I'], sys_set['NU_I'], y, function=f_ion, kappa=kappa)

    # Electrons
    params = {'THETA': sys_set['THETA'], 'nu': sys_set['NU_E'], 'm': const.m_e, 'T': sys_set['T_E'], 'T_ES': sys_set['T_ES'],
              'w_c': w_c, 'kappa': kappa, 'vdf': vdf, 'Z': sys_set['Z'], 'mat_file': sys_set['mat_file']}
    y = np.linspace(0, cf.Y_MAX_e**(1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double)**cf.ORDER
    func.initialize(y, params)
    Fe = para.integrate(
        const.m_e, sys_set['T_E'], sys_set['NU_E'], y, function=func, kappa=kappa)

    Xp_i = np.sqrt(
        1 / (2 * L_Debye(sys_set['NE'], sys_set['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))
    Xp_e = np.sqrt(
        1 / (2 * L_Debye(sys_set['NE'], sys_set['T_E'], kappa=kappa)**2 * cf.K_RADAR**2))

    f_scaled = cf.f
    # In case we have \omega = 0 in our frequency array, we just ignore this warning message
    with np.errstate(divide='ignore', invalid='ignore'):
        Is = sys_set['NE'] / (np.pi * cf.w) * (np.imag(- Fe) * abs(1 + 2 * Xp_i**2 * Fi)**2 + (
            4 * Xp_e**4 * np.imag(- Fi) * abs(Fe)**2)) / abs(1 + 2 * Xp_e**2 * Fe + 2 * Xp_i**2 * Fi)**2

    if area:
        if cf.I_P['F_MAX'] < 1e4:
            area = si.simps(Is, cf.f)
            print('The area under the ion line is %1.6e.' % area)
        else:
            print('F_MAX is set too high. The area was not calculated.')

    sys_set['THETA'] = round(params['THETA'] * 180 / np.pi, 1)
    return f_scaled, Is, dict(sys_set, **p)


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


def correct_inputs(version, sys_set, params):
    """Extra check suppressing the parameters that was given but is not necessary.
    """
    if version != 'kappa' and not (version == 'long_calc' and params['vdf'] in ['kappa', 'kappa_vol2']):
        params['kappa'] = None
    if version != 'long_calc':
        params['vdf'] = None
    if version != 'long_calc' or params['vdf'] != 'gauss_shell':
        sys_set['T_ES'] = None
    if version != 'long_calc' or params['vdf'] != 'real_data':
        sys_set['Z'] = None
        sys_set['mat_file'] = None
    return sys_set, params


def version_check(version, vdf, kappa, sys_set):
    versions = ['kappa', 'maxwell', 'long_calc']
    try:
        if not version in versions:
            raise SystemError
        print(f'Using version "{version}"', flush=True)
    except SystemError:
        sys.exit(version_error(version, versions))
    if version == 'maxwell':
        func = intf.INT_MAXWELL()
    elif version == 'kappa':
        kappa_check(kappa, sys_set)
        func = intf.INT_KAPPA()
    elif version == 'long_calc':
        vdfs = ['maxwell', 'kappa', 'kappa_vol2', 'gauss_shell', 'real_data']
        try:
            if not vdf in vdfs:
                raise SystemError
            print(f'Using VDF "{vdf}"', flush=True)
        except Exception:
            sys.exit(version_error(vdf, vdfs, element='VDF'))
        if vdf in ['kappa', 'kappa_vol2']:
            kappa_check(kappa, sys_set)
            if isinstance(kappa, list):
                sys.exit(print('kappa as a list is not accepted for the long_calc version.'))
        func = intf.INT_LONG()
    return func


def version_error(version, versions, element='version'):
    exc_type, _, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f'{exc_type} error in file {fname}, line {exc_tb.tb_lineno}')
    print(f'The {element} is wrong: "{version}" not found in {versions}')


def kappa_check(kappa, sys_set):
    try:
        if kappa is None:
            raise SystemError
    except SystemError:
        sys.exit(print('You forgot to send in the kappa parameter.'))
    if sys_set['NU_E'] != 0 or sys_set['NU_I'] != 0:
        text = f'''\
                Warning: the kappa function is defined for a collisionless plasma.
                You are using: nu_i = {sys_set['NU_I']} and nu_e = {sys_set['NU_E']}.'''
        print(txt.fill(txt.dedent(text), width=300))
