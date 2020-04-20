"""Main script for calculating the IS spectrum.
"""

import os
import time
import datetime
# The start method of the multiprocessing module was changed from python3.7 to python3.8.
# Instead of using 'fork', 'spawn' is the new default. To be able to use global
# variables across all parallel processes, the start method must be reset to 'fork'.
# See https://docs.python.org/3/library/multiprocessing.html for more info.
import multiprocessing as mp
mp.set_start_method('fork')

import si_prefix as sip  # pylint: disable=C0413
import matplotlib.pyplot as plt  # pylint: disable=C0413
from matplotlib.backends.backend_pdf import PdfPages  # pylint: disable=C0413
import matplotlib.gridspec as grid_spec
# import seaborn as sns
import numpy as np  # pylint: disable=C0413
import scipy.constants as const  # pylint: disable=C0413
# import joypy

from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413



def scale_f(frequency):
    """Scale the axis and add the appropriate SI prefix.

    Arguments:
        frequency {np.ndarray} -- the variable along an axis

    Returns:
        str, np.ndarray, int -- the prefix, the scaled variables, the exponent corresponding to the prefix
    """
    freq = np.copy(frequency)
    exp = sip.split(np.max(freq))[1]
    freq /= 10**exp
    pre = sip.prefix(exp)
    return pre, freq, exp


def find_p_line(freq, spec, scale):
    """Find the frequency that is most likely the peak of the plasma line
    and return the lower and upper bounds for an interval around the peak.

    Arguments:
        freq {np.ndarray} -- sample points of frequency parameter
        spec {np.ndarray} -- values of spectrum at the sampled frequencies
        scale {int} -- exponent corresponding to the prefix of the frequency scale

    Returns:
        float, float -- lower and upper bound of the interval
    """
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


def ridge_plot(version, kappa=None, vdf=None, area=False, plasma=False, info=None):  # f, spectrum
    # joypy.joyplot()
    TEMPS = cf.I_P['T_E'].copy()
    TEMPS.reverse()
    gs = grid_spec.GridSpec(len(TEMPS), 1)
    fig = plt.figure()
    i = 0
    ax_objs = []
    Rgb, rGb, rgB = 0.0, 0.0, 1.0
    gradient = 1 / len(TEMPS)
    for TEMP in TEMPS:
        cf.I_P['T_E'] = TEMP
        f, s = isr.isr_spectrum(version, kappa=kappa, area=area, vdf=vdf)
        # plt.subplot(len(spectrum), 1, i)
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax_objs[-1].plot(f, s, color='w', linewidth=1)
        ax_objs[-1].fill_between(f, s, alpha=1, color=(Rgb, rGb, rgB))
        Rgb += gradient
        rgB -= gradient

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if i == len(TEMPS) - 1:
            pass
        else:
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        spines = ["top","right","left","bottom"]
        for sp in spines:
            ax_objs[-1].spines[sp].set_visible(False)
        ax_objs[-1].text(f[0], np.max(s) * .2, r'$T_e$=' + f'{TEMP}',
                         fontsize=14, ha="right")

        i += 1

    gs.update(hspace=-0.4)
    # plt.tight_layout()
    plt.show()


def plotter(f, Is, plot_func, l=None, plasma=False):
    """Plot the spectrum using given plot method.

    Arguments:
        f {np.ndarray} -- sample points in frequency
        Is {np.ndarray} -- spectrum values at sampled frequency
        plot_func {function} -- plotting function from matplotlib.pyplot

    Keyword Arguments:
        l {float or list of floats} -- kappa index (default: {None})
        plasma {bool} -- wether to plot only the plasma line of the spectrum or not (default: {False})
    """
    Is = Is.copy()
    if plot_func == plt.plot:  # pylint: disable=W0143
        idx = np.argwhere(abs(f) < 4e4)
        f = f[idx].reshape((-1,))
        if isinstance(Is, list):
            for i, _ in enumerate(Is):
                Is[i] = Is[i][idx].reshape((-1,))
        else:
            Is = Is[idx].reshape((-1,))
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
    if plot_func == plt.semilogy:  # pylint: disable=W0143
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


def saver(f, Is, version, kappa=None, plasma=False, vdf=None, info=None):
    """Save the generated plots to a single pdf file.

    Arguments:
        f {np.ndarray} -- sample points in frequency
        Is {np.ndarray} -- values of the spectrum at the sampled frequencies
        version {str} -- the version used to calculate the spectrum

    Keyword Arguments:
        kappa {float or list of floats} -- kappa index (default: {None})
        plasma {bool} -- wether to plot only the plasma line of the spectrum or not (default: {False})
        vdf {str} -- the VDF used in the long_calc version (default: {None})
        info {str} -- extra information for the pdf metadata (default: {None})
    """
    cf.I_P['THETA'] = round(cf.I_P['THETA'] * 180 / np.pi, 1)
    if info is None:
        I_P = dict({'vdf': vdf, 'kappa': kappa, 'F_N_POINTS': cf.F_N_POINTS,
                    'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS}, **cf.I_P)
    else:
        I_P = dict({'vdf': vdf, 'kappa': kappa, 'info': info, 'F_N_POINTS': cf.F_N_POINTS,
                    'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS}, **cf.I_P)
    tt = time.localtime()
    the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
    os.makedirs('../../../report/master-thesis/figures', exist_ok=True)
    pdffig = PdfPages(
        f'../../../report/master-thesis/figures/{the_time}_{version}.pdf')
    metadata = pdffig.infodict()
    metadata['Title'] = f'ISR Spectrum w/ {version}'
    metadata['Author'] = 'Eirik R. Enger'
    metadata['Subject'] = f"IS spectrum made using a {version} distribution and Simpson's integration rule."
    metadata['Keywords'] = f'{I_P}'
    metadata['ModDate'] = datetime.datetime.today()
    plotter(f, Is, plt.semilogy, kappa, plasma=plasma)
    pdffig.attach_note("Semilog y")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    plotter(f, Is, plt.plot, kappa, plasma=plasma)
    pdffig.attach_note("Linear plot")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    plotter(f, Is, plt.loglog, kappa, plasma=plasma)
    pdffig.attach_note("Loglog")
    plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
    # plotter(f, Is, plt.semilogx, l, plasma=plasma)
    pdffig.close()


def plot_IS_spectrum(version, kappa=None, vdf=None, area=False, plasma=False, info=None):
    """Make plots of an IS spectrum based on a variety of VDFs.

    Arguments:
        version {str} -- chose to the quicker maxwell or kappa version (analytic solution) or
                         the long_calc for arbitrary isotropic VDF (numerical solution only)

    Keyword Arguments:
        kappa {int or float} -- kappa index for the kappa VDFs (default: {None})
        vdf {str} -- when using 'long_calc', set which VDF to use (default: {None})
        area {bool} -- if the spectrum is small enough around zero frequency, calculate the area (under the ion line) (default: {False})
        plasma {bool} -- choose to plot only the part of the spectrum where the plasma line is found (default: {False})
        info {str} -- optional extra info that will be saved to the pdf metadata made for the plots (default: {None})
    """
    save = input(
        'Press "y/yes" to save plot, any other key to dismiss.\t'
    ).lower()
    spectrum = False
    if isinstance(kappa, list) and version == 'kappa':
        spectrum = []
        f, Is = isr.isr_spectrum('maxwell', area=area)
        spectrum.append(Is)
        for k in kappa:
            f, Is = isr.isr_spectrum('kappa', kappa=k, area=area)
            spectrum.append(Is)
    elif isinstance(kappa, int) and version == 'kappa':
        f, Is = isr.isr_spectrum('kappa', kappa=kappa, area=area)
    else:
        f, Is = isr.isr_spectrum(version, kappa=kappa, area=area, vdf=vdf)
    # ridge_plot(f, spectrum)
    # fig, axes = joypy.joyplot(spectrum, fill=False)
    if spectrum:
        if save in ['y', 'yes']:
            saver(f, spectrum, 'both', kappa=kappa, plasma=plasma, vdf=vdf)
        else:
            plotter(f, spectrum, plt.plot, l=kappa, plasma=plasma)
            plotter(f, spectrum, plt.semilogy, l=kappa, plasma=plasma)
            # plotter(f, spectrum, plt.semilogx, l=kappa, plasma=plasma)
            plotter(f, spectrum, plt.loglog, l=kappa, plasma=plasma)
    else:
        if save in ['y', 'yes']:
            saver(f, Is, version, kappa=kappa,
                  plasma=plasma, info=info, vdf=vdf)
        else:
            plotter(f, Is, plt.plot, plasma=plasma)
            plotter(f, Is, plt.semilogy, plasma=plasma)
            # plotter(f, Is, plt.semilogx, plasma=plasma)
            plotter(f, Is, plt.loglog, plasma=plasma)
    plt.show()


if __name__ == '__main__':
    version = 'kappa'
    kwargs = {'vdf': 'gauss_shell', 'info': 'with Maxwellian ions', 'kappa': 5}
    if isinstance(cf.I_P['T_E'], list):
        ridge_plot(version, **kwargs)
    else:
        plot_IS_spectrum(version, **kwargs)
