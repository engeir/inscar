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


def find_p_line(freq, spec, scale):
    fr = np.copy(freq)
    sp = np.copy(spec)
    w_p = np.sqrt(cf.I_P['NE'] * const.elementary_charge **
                  2 / (const.m_e * const.epsilon_0))
    f = w_p * (1 + 3 * cf.K_RADAR**2 *
               cf.I_P['T_E'] * const.k / (const.m_e * w_p**2))**.5 / (2 * np.pi)
    lower, upper = (f - 1e6) / 10**scale, (f + 1e6) / 10**scale
    m = (fr > lower) & (fr < upper)
    fr_n = fr[m]
    sp = sp[m]
    av = fr_n[np.argmax(sp)]
    lower, upper = av - 2e6 / 10**scale, av + 2e6 / 10**scale
    return lower, upper


def plotter(f, Is, plot_func, l=None, plasma=False):
    p, freq, exp = scale_f(f)
    plt.figure()
    if plasma:
        if isinstance(Is, list):
            spectrum = Is[0]
        else:
            spectrum = Is
        mini, maxi = find_p_line(freq, spectrum, exp)
        mask = (freq > mini) & (freq < maxi)
        freq = freq[mask]
    if plot_func == plt.semilogy:
        plt.xlabel(f'Frequency [{p}Hz]')
        plt.ylabel('10*log10(Power) [dB]')
        plot_func = plt.plot
        if isinstance(Is, list):
            for s in Is:
                s = 10 * np.log10(s)
        else:
            Is = 10 * np.log10(Is)
    else:
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
            plot_func(freq, s, 'r', linestyle=st, linewidth=.8, label=lab)
        plt.legend()
    else:
        if plasma:
            Is = Is[mask]
        plot_func(freq, Is, 'r')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def saver(f, Is, version, l=None, kappa=None, plasma=False):
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
    plotter(f, Is, plt.semilogy, l, plasma=plasma)
    pdffig.attach_note("Semilog y")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    plotter(f, Is, plt.plot, l, plasma=plasma)
    pdffig.attach_note("Linear plot")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    plotter(f, Is, plt.loglog, l, plasma=plasma)
    pdffig.attach_note("Loglog")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    # plotter(f, Is, plt.semilogx, l, plasma=plasma)
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
            plotter(f, spectrum, plt.plot, l=kappa, plasma=plasma)
            plotter(f, spectrum, plt.semilogy, l=kappa, plasma=plasma)
            # plotter(f, spectrum, plt.semilogx, l=kappa, plasma=plasma)
            plotter(f, spectrum, plt.loglog, l=kappa, plasma=plasma)
    else:
        if save in ['y', 'yes']:
            if version == 'kappa':
                saver(f, Is, version, kappa=kappa, plasma=plasma)
            else:
                saver(f, Is, version, plasma=plasma)
        else:
            plotter(f, Is, plt.plot, plasma=plasma)
            plotter(f, Is, plt.semilogy, plasma=plasma)
            # plotter(f, Is, plt.semilogx, plasma=plasma)
            plotter(f, Is, plt.loglog, plasma=plasma)
    plt.show()


if __name__ == '__main__':
    # TODO: when both functions are run using the same version, we do not need to calculate Fe and Fi twice.
    plot_IS_spectrum('long_calc')  # , kappa=3, area=True, plasma=False)
    # tool.H_spectrum('kappa')
