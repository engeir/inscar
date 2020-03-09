"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import os
import time
import datetime

import si_prefix as sip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.constants as const

import config as cf
import tool


def scale_f(frequency):
    freq = np.copy(frequency)
    exp = sip.split(np.max(freq))[1]
    freq /= 10**exp
    pre = sip.prefix(exp)
    return pre, freq, exp


def find_p_line(scale):
    w_p = np.sqrt(cf.I_P['NE'] * const.elementary_charge**2 / (const.m_e * const.epsilon_0))
    f = w_p * (1 + 3 * cf.K_RADAR**2 *
               cf.I_P['T_E'] * const.k / (const.m_e * w_p**2))**.5 / (2 * np.pi)
    return (f - 3e5) / 10**scale, (f + 3e5) / 10**scale


def loglog(f, Is, l=None, plasma=False):
    p, freq, exp = scale_f(f)
    plt.figure()
    # plt.title('ISR spectrum')
    if plasma:
        mini, maxi = find_p_line(exp)
        mask = (freq > mini) & (freq < maxi)
        freq = freq[mask]
    plt.xlabel(f'Frequency [{p}Hz]')
    plt.ylabel('Power')
    if isinstance(Is, list):
        if any([isinstance(i, int) for i in l]):
            for v, i in enumerate(l):
                l[v] = r'$\kappa =$' + f'{i}'
        if 'Maxwellian' not in l:
            l.insert(0, 'Maxwellian')
        style = ['-', '--', ':', '-.',
                 (0, (3, 5, 1, 5, 1, 5)),
                 (0, (3, 1, 1, 1, 1, 1))]
        for st, s, lab in zip(style, Is, l):
            if plasma:
                s = s[mask]
            plt.loglog(freq, s, 'r', linestyle=st, linewidth=.8, label=lab)
        plt.legend()
    else:
        if plasma:
            Is = Is[mask]
        plt.loglog(freq, Is, 'r')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_y(f, Is, l=None, plasma=False):
    p, freq, exp = scale_f(f)
    plt.figure()
    # plt.title('ISR spectrum')
    if plasma:
        mini, maxi = find_p_line(exp)
        mask = (freq > mini) & (freq < maxi)
        freq = freq[mask]
    plt.xlabel(f'Frequency [{p}Hz]')
    plt.ylabel('10*log10(Power) [dB]')
    # plt.semilogy(f, Is, 'r')
    if isinstance(Is, list):
        if any([isinstance(i, int) for i in l]):
            for v, i in enumerate(l):
                l[v] = r'$\kappa =$' + f'{i}'
        if 'Maxwellian' not in l:
            l.insert(0, 'Maxwellian')
        style = ['-', '--', ':', '-.',
                 (0, (3, 5, 1, 5, 1, 5)),
                 (0, (3, 1, 1, 1, 1, 1))]
        for st, s, lab in zip(style, Is, l):
            if plasma:
                s = s[mask]
            plt.plot(freq, 10 * np.log10(s), 'r',
                     linestyle=st, linewidth=.8, label=lab)
        plt.legend()
    else:
        if plasma:
            Is = Is[mask]
        plt.plot(freq, 10 * np.log10(Is), 'r')
    # plt.yscale('log')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_x(f, Is, l=None, plasma=False):
    p, freq, exp = scale_f(f)
    plt.figure()
    # plt.title('ISR spectrum')
    if plasma:
        mini, maxi = find_p_line(exp)
        mask = (freq > mini) & (freq < maxi)
        freq = freq[mask]
    plt.xlabel(f'Frequency [{p}Hz]')
    plt.ylabel('Power')
    if isinstance(Is, list):
        if any([isinstance(i, int) for i in l]):
            for v, i in enumerate(l):
                l[v] = r'$\kappa =$' + f'{i}'
        if 'Maxwellian' not in l:
            l.insert(0, 'Maxwellian')
        style = ['-', '--', ':', '-.',
                 (0, (3, 5, 1, 5, 1, 5)),
                 (0, (3, 1, 1, 1, 1, 1))]
        for st, s, lab in zip(style, Is, l):
            if plasma:
                s = s[mask]
            plt.semilogx(freq, s, 'r', linestyle=st, linewidth=.8, label=lab)
        plt.legend()
    else:
        if plasma:
            Is = Is[mask]
        plt.semilogx(freq, Is, 'r')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def two_side_lin_plot(f, Is, l=None, plasma=False):
    p, freq, exp = scale_f(f)
    plt.figure()
    # plt.title('ISR spectrum')
    if plasma:
        mini, maxi = find_p_line(exp)
        mask = (freq > mini) & (freq < maxi)
        freq = freq[mask]
    plt.xlabel(f'Frequency [{p}Hz]')
    plt.ylabel('Power')
    if isinstance(Is, list):
        if any([isinstance(i, int) for i in l]):
            for v, i in enumerate(l):
                l[v] = r'$\kappa =$' + f'{i}'
        if 'Maxwellian' not in l:
            l.insert(0, 'Maxwellian')
        style = ['-', '--', ':', '-.',
                 (0, (3, 5, 1, 5, 1, 5)),
                 (0, (3, 1, 1, 1, 1, 1))]
        for st, s, lab in zip(style, Is, l):
            if plasma:
                s = s[mask]
            plt.plot(freq, s, 'r', linestyle=st, linewidth=.8, label=lab)
        plt.legend()
    else:
        if plasma:
            Is = Is[mask]
        plt.plot(freq, Is, 'r')
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.tight_layout()


def saver(f, Is, version, l=None, kappa=None, plasma=False):
    # I_P = dict(cf.I_P, **{'F_N_POINTS': cf.F_N_POINTS,
    #                       'N_POINTS': cf.N_POINTS})
    I_P = dict(cf.I_P, **{'kappa': kappa,
                          'F_N_POINTS': cf.F_N_POINTS, 'N_POINTS': cf.N_POINTS})
    tt = time.localtime()
    the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
    pdffig = PdfPages(
        f'../../report/master-thesis/figures/{the_time}_{version}.pdf')
    os.makedirs('../../report/master-thesis/figures', exist_ok=True)
    metadata = pdffig.infodict()
    metadata['Title'] = f'ISR Spectrum w/ {version}'
    metadata['Author'] = 'Eirik R. Enger'
    metadata['Subject'] = f"IS spectrum made using a {version} distribution and Simpson's integration rule."
    metadata['Keywords'] = f'{I_P}'
    metadata['ModDate'] = datetime.datetime.today()
    semilog_y(f, Is, l, plasma=plasma)
    pdffig.attach_note("Semilog y")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    two_side_lin_plot(f, Is, l, plasma=plasma)
    pdffig.attach_note("Linear plot")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    loglog(f, Is, l, plasma=plasma)
    pdffig.attach_note("Loglog")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    # semilog_x(f, Is, l, plasma=plasma)
    pdffig.close()


def plot_IS_spectrum(version, kappa=None, area=False, plasma=False):
    save = input(
        'Press "y/yes" to save plot, any other key to dismiss.\t').lower()
    spectrum = False
    if isinstance(kappa, list) and version == 'kappa':
        spectrum = []
        f, Is = tool.isr_spectrum('maxwell', area=area)
        spectrum.append(Is)
        for k in kappa:
            f, Is = tool.isr_spectrum('kappa', kappa=k, area=area)
            spectrum.append(Is)
    elif isinstance(kappa, int) and version == 'kappa':
        f, Is = tool.isr_spectrum('kappa', kappa=kappa)
    else:
        f, Is = tool.isr_spectrum(version)
    if spectrum:
        if save in ['y', 'yes']:
            saver(f, spectrum, 'both', l=kappa, kappa=kappa, plasma=plasma)
        else:
            two_side_lin_plot(f, spectrum, l=kappa, plasma=plasma)
            loglog(f, spectrum, l=kappa, plasma=plasma)
            # semilog_x(f, spectrum, l=kappa, plasma=plasma)
            semilog_y(f, spectrum, l=kappa, plasma=plasma)
    else:
        if save in ['y', 'yes']:
            if version == 'kappa':
                saver(f, Is, version, kappa=kappa, plasma=plasma)
            else:
                saver(f, Is, version, plasma=plasma)
        else:
            two_side_lin_plot(f, Is, plasma=plasma)
            loglog(f, Is, plasma=plasma)
            # semilog_x(f, Is, plasma=plasma)
            semilog_y(f, Is, plasma=plasma)
    plt.show()


if __name__ == '__main__':
    # TODO: when both functions are run using the same version, we do not need to calculate Fe and Fi twice.
    plot_IS_spectrum('kappa', kappa=[3, 10], area=True, plasma=True)
    # tool.H_spectrum('kappa')
