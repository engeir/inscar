"""Main script for  controlling the calculation method of the IS spectrum.
"""

import os
import sys
import time
import datetime
# The start method of the multiprocessing module was changed from python3.7 to python3.8.
# Instead of using 'fork', 'spawn' is the new default. To be able to use global
# variables across all parallel processes, the start method must be reset to 'fork'.
# See https://docs.python.org/3/library/multiprocessing.html for more info.
import multiprocessing as mp
mp.set_start_method('fork')

import matplotlib  # pylint: disable=C0413
import matplotlib.gridspec as grid_spec  # pylint: disable=C0413
import matplotlib.pyplot as plt  # pylint: disable=C0413
from matplotlib.backends.backend_pdf import PdfPages  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413
import scipy.constants as const  # pylint: disable=C0413
import si_prefix as sip  # pylint: disable=C0413

from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413

# From https://stackoverflow.com/questions/47253462/matplotlib-2-mathtext-glyph-errors-in-tick-labels
# Customize matplotlib
matplotlib.rcParams.update({  # Use mathtext, not LaTeX
    'text.usetex': False,
    'font.family': 'Ovo',
    'font.serif': 'Ovo',
    'mathtext.fontset': 'cm',
    # Use ASCII minus
    'axes.unicode_minus': False,
})


class CreateData:
    """Creation class with methods that return frequency range and spectra.
    """

    def __init__(self, version, kappa=None, vdf=None, area=False):
        """Initialize the class with correct variables.

        Arguments:
            version {str} -- chose to use the quicker maxwell or kappa versions (analytic solution) or
                             the long_calc for arbitrary isotropic VDF (numerical solution only)

        Keyword Arguments:
            kappa {int or float} -- kappa index for the kappa VDFs (default: {None})
            vdf {str} -- when using 'long_calc', set which VDF to use (default: {None})
            area {bool} -- if the spectrum is small enough around zero frequency, calculate the area (under the ion line) (default: {False})
        """
        self.version = version
        self.kappa = kappa
        self.vdf = vdf
        self.area = area

    def create_single_spectrum(self):
        """Create one spectrum only.

        Returns:
            np.ndarray -- two arrays, one for frequency and one for the corresponding power
        """
        f, s = isr.isr_spectrum(
            self.version, vdf=self.vdf, kappa=self.kappa, area=self.area)
        return f, s

    def create_multi_spectrum(self):
        """Create a list of spectra, one for each kappa index, plus one maxwellian for reference.

        Returns:
            np.ndarray and list of np.ndarrays -- frequency and a list of corresponding spectra
        """
        multi_spectrum = []
        f, s = isr.isr_spectrum('maxwell', area=self.area)
        multi_spectrum.append(s)
        for k in self.kappa:
            _, s = isr.isr_spectrum(
                self.version, vdf=self.vdf, kappa=k, area=self.area)
            multi_spectrum.append(s)
        return f, multi_spectrum

    def create_single_or_multi(self):
        """Create spectra for one set of plasma parameters.

        The method automatically decides between single or multi spectrum.

        Returns:
            np.ndarray x2 or np.ndarray and list of np.ndarrays -- frequency and list of spectra or only one spectrum
        """
        if isinstance(self.kappa, list):
            f, s = self.create_multi_spectrum()
        else:
            f, s = self.create_single_spectrum()
        return f, s

    def create_multi_params(self):
        """Create multiple spectra for different set of plasma parameters.

        For each set of plasma parameters, either single or multi spectra can be created.

        Returns:
            np.ndarray and list of list (of np.ndarrays) or np.ndarray -- returns frequency and resulting spectra from single_or_multi
        """
        I_P_original = cf.I_P.copy()
        I_P_copy = cf.I_P.copy()
        for item in I_P_copy:
            try:
                item.reverse()
            except Exception:
                pass
        multi_params = []
        for temp in I_P_copy['T_E']:
            cf.I_P['T_E'] = temp
            f, s = self.create_single_or_multi()
            multi_params.append(s)
        cf.I_P = I_P_original
        return f, multi_params


class PlotClass:
    """Create a plot object that automatically will show the data created.
    """

    def __init__(self, version, kappa=None, vdf=None, area=False, plasma=False, info=None):
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
        self.version = version
        self.kappa = kappa
        self.vdf = vdf
        self.plasma = plasma
        self.info = info
        self.correct_inputs()
        self.create_data = CreateData(self.version, self.kappa, self.vdf, area)
        save = input(
            'Press "y/yes" to save plot, any other key to dismiss.\t').lower()
        self.setup()
        self.final(save)

    def correct_inputs(self):
        """Extra check suppressing the parameters that was given but is not necessary.
        """
        if not self.version == 'kappa' and not (self.version == 'long_calc' and self.vdf in ['kappa', 'kappa_vol2']):
            self.kappa = None
        if not self.version == 'long_calc':
            self.vdf = None
        if not self.vdf == 'gauss_shell':
            cf.I_P['T_ES'] = None
        if self.plasma:
            if isinstance(cf.I_P['T_E'], list):
                T_0 = cf.I_P['T_E'][0]
            else:
                T_0 = cf.I_P['T_E']
            if self.find_p_line(None, None, None, T_0, check=True):
                sys.exit(print(f"F_MAX (= {cf.I_P['F_MAX']}) is not high enough to look at the plasma line."))

    def setup(self):
        """Do initial tasks. Decide on what kind of plot and create correct data.
        """
        if isinstance(self.kappa, list):
            self.version = 'both'
        if isinstance(cf.I_P['T_E'], list):
            self.plot_type = 'ridge'
            self.f, self.data = self.create_data.create_multi_params()
        else:
            self.plot_type = 'normal'
            self.f, self.data = self.create_data.create_single_or_multi()

    def final(self, save):
        """Make the plots from the created data and save if needed.

        Arguments:
            save {str} -- if 'y' or 'yes', the figure is saved to a predefined directory
        """
        if save in ['y', 'yes']:
            self.save_me()
        else:
            self.plot('semilogy')
            self.plot('loglog')
            self.plot('plot')
        plt.show()

    def save_me(self):
        """Save the figure as a multi page pdf with all parameters saved in the meta data.

        The date and time is used in the figure name, in addition to it ending with which method was used.
        The settings that was used in config as as inputs to the plot object is saved in the metadata of the figure.
        """
        cf.I_P['THETA'] = round(cf.I_P['THETA'] * 180 / np.pi, 1)
        if self.info is None:
            I_P = dict({'vdf': self.vdf, 'kappa': self.kappa, 'F_N_POINTS': cf.F_N_POINTS,
                        'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS}, **cf.I_P)
        else:
            I_P = dict({'vdf': self.vdf, 'kappa': self.kappa, 'info': self.info, 'F_N_POINTS': cf.F_N_POINTS,
                        'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS}, **cf.I_P)
        tt = time.localtime()
        the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
        os.makedirs('../../../report/master-thesis/figures', exist_ok=True)
        pdffig = PdfPages(
            f'../../../report/master-thesis/figures/{the_time}_{self.version}.pdf')
        metadata = pdffig.infodict()
        metadata['Title'] = f'ISR Spectrum w/ {self.version}'
        metadata['Author'] = 'Eirik R. Enger'
        metadata['Subject'] = f"IS spectrum made using a {self.version} distribution and Simpson's integration rule."
        metadata['Keywords'] = f'{I_P}'
        metadata['ModDate'] = datetime.datetime.today()
        self.plot('semilogy')
        pdffig.attach_note("Semilog y")
        plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
        self.plot('plot')
        pdffig.attach_note("Linear plot")
        plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
        self.plot('loglog')
        pdffig.attach_note("Loglog")
        plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
        pdffig.close()

    def plot(self, func_type):
        """Make a plot independent of what kind of data is used, and with any given plotting method.

        Arguments:
            func_type {str} -- any attribute of the matplotlib.pyplot object is accepted and is used to do the plotting
        """
        try:
            getattr(plt, func_type)
        except Exception:
            print(f'{func_type} is not an attribute of the matplotlib.pyplot object. Skips to next.')
        else:
            if self.plot_type == 'ridge':
                self.plot_ridge(self.f, self.data, func_type)
            else:
                self.plot_normal(self.f, self.data, func_type)

    def plot_normal(self, f, Is, func_type):
        """Make a plot using f as x-axis scale and Is as values.

        Is may be either a (N,) or (N,1) np.ndarray or a list of such arrays.

        Arguments:
            f {np.ndarray} -- variable along x-axis
            Is {np.ndarray or list} -- y-axis values along x-axis
            func_type {str} -- attribute of the matplotlib.pyplot object
        """
        Is = Is.copy()
        # Linear plot show only ion line (kHz range).
        if func_type == 'plot' and not self.plasma:
            f, Is = self.only_ionline(f, Is)
        p, freq, exp = self.scale_f(f)
        plt.figure(figsize=(6, 3))
        if self.plasma:
            # Clip the frequency line around the plasma frequency.
            if isinstance(Is, list):
                spectrum = Is[0]
            else:
                spectrum = Is
            mini, maxi = self.find_p_line(freq, spectrum, exp, cf.I_P['T_E'])
            mask = (freq > mini) & (freq < maxi)
            freq = freq[mask]
        if func_type == 'semilogy':
            # Rescale the y-axis to a dB scale.
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel(
                '10*log10(Power) [dB]')
            if isinstance(Is, list):
                for i, _ in enumerate(Is):
                    Is[i] = 10 * np.log10(Is[i])
            else:
                Is = 10 * np.log10(Is)
        else:
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel('Power')
        if isinstance(Is, list):
            if any([isinstance(i, int) for i in self.kappa]):
                for v, i in enumerate(self.kappa):
                    self.kappa[v] = r'$\kappa =$' + f'{i}'
            if 'Maxwellian' not in self.kappa:
                self.kappa.insert(0, 'Maxwellian')
            style = ['-', '--', ':', '-.',
                     (0, (3, 5, 1, 5, 1, 5)),
                     (0, (3, 1, 1, 1, 1, 1))]
            for st, s, lab in zip(style, Is, self.kappa):
                if self.plasma:
                    s = s[mask]
                if func_type == 'semilogy':
                    plt.plot(freq, s, 'r', linestyle=st,
                             linewidth=.8, label=lab)
                else:
                    plot_object = getattr(plt, func_type)
                    plot_object(freq, s, 'r', linestyle=st,
                                linewidth=.8, label=lab)
            plt.legend()
        else:
            if self.plasma:
                Is = Is[mask]
            if func_type == 'semilogy':
                plt.plot(freq, Is, 'r')
            else:
                plot_object = getattr(plt, func_type)
                plot_object(freq, Is, 'r')
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.tight_layout()

    def plot_ridge(self, f, multi_parameters, func_type):
        # Inspired by https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
        # To make sure not to alter the object in the config file, it is copied.
        multi_params = multi_parameters.copy()
        multi_params.reverse()
        TEMPS = cf.I_P['T_E'].copy()
        TEMP_0 = TEMPS[0]
        TEMPS.reverse()
        gs = grid_spec.GridSpec(len(multi_params), 1)
        fig = plt.figure(figsize=(7, 9))
        ax_objs = []
        Rgb, rGb, rgB = 0.0, 0.0, 1.0
        gradient = 1 / (len(multi_params) - .95)
        f_original = f.copy()
        # If you want equal scaling of the y axis as well
        # y_min, y_max = self.scaling_y(multi_params)
        for j, params in enumerate(multi_params):
            # f is reset due to the scaling immediately below.
            f = f_original
            # Linear plot show only ion line (kHz range).
            if func_type == 'plot' and not self.plasma:
                f, params = self.only_ionline(f, params)
            p, freq, exp = self.scale_f(f)
            if self.plasma:
                if isinstance(params, list):
                    spectrum = params[0]
                else:
                    spectrum = params
                mini, maxi = self.find_p_line(freq, spectrum, exp, temp=TEMP_0)
                mask = (freq > mini) & (freq < maxi)
                freq = freq[mask]
            ax_objs.append(fig.add_subplot(gs[j:j + 1, 0:]))
            if isinstance(params, list):
                if any([isinstance(kappa_i, int) for kappa_i in self.kappa]):
                    for v, kappa_i in enumerate(self.kappa):
                        self.kappa[v] = r'$\kappa =$' + f'{kappa_i}'
                if 'Maxwellian' not in self.kappa:
                    self.kappa.insert(0, 'Maxwellian')
                style = ['-', '--', ':', '-.',
                         (0, (3, 5, 1, 5, 1, 5)),
                         (0, (3, 1, 1, 1, 1, 1))]
                first = 0
                for st, s, lab in zip(style, params, self.kappa):
                    if self.plasma:
                        s = s[mask]
                    plot_object = getattr(ax_objs[-1], func_type)
                    plot_object(freq, s, color=(Rgb, rGb, rgB), linewidth=1, label=lab, linestyle=st)
                    if first == 0:
                        x_0 = ax_objs[-1].viewLim.x0
                        idx = np.argwhere(freq > x_0)[0]
                        x1, y1 = ax_objs[-1].viewLim.x1, np.max(s)
                        ax_objs[-1].text(freq[idx], s[idx], r'$T_e$ = ' + f'{TEMPS[j]} K',
                                         fontsize=14, ha="right", va='bottom')
                    # ax_objs[-1].fill_between(freq, s, alpha=1, color=(Rgb, rGb, rgB))
                    if j == 0:
                        plt.legend(loc='upper right', bbox_to_anchor=(x1, y1), bbox_transform=ax_objs[-1].transData)
                    first += 1
            else:
                if self.plasma:
                    params = params[mask]
                plot_object = getattr(ax_objs[-1], func_type)
                plot_object(freq, params, color=(Rgb, rGb, rgB), linewidth=1)
                x_0 = ax_objs[-1].viewLim.x0
                idx = np.argwhere(freq > x_0)[0]
                ax_objs[-1].text(freq[idx], params[idx], r'$T_e$ = ' + f'{TEMPS[j]} K',
                                 fontsize=14, ha="right", va='bottom')
                # ax_objs[-1].fill_between(freq, params, alpha=1, color=(Rgb, rGb, rgB))
            Rgb += gradient
            rgB -= gradient

            # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)

            # remove borders, axis ticks and labels
            # plt.ylim([y_min, y_max])
            ax_objs[-1].set_yticklabels([])
            plt.tick_params(axis='y', which='both', left=False,
                            right=False, labelleft=False)
            if j == len(multi_params) - 1:
                plt.xlabel(f'Frequency [{p}Hz]')
            else:
                plt.tick_params(axis='x', which='both', bottom=False,
                                top=False, labelbottom=False)

            spines = ["top", "right", "left", "bottom"]
            for sp in spines:
                ax_objs[-1].spines[sp].set_visible(False)

        gs.update(hspace=-0.6)

    @staticmethod
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

    @staticmethod
    def find_p_line(freq, spec, scale, temp, check=False):
        """Find the frequency that is most likely the peak of the plasma line
        and return the lower and upper bounds for an interval around the peak.

        Arguments:
            freq {np.ndarray} -- sample points of frequency parameter
            spec {np.ndarray} -- values of spectrum at the sampled frequencies
            scale {int} -- exponent corresponding to the prefix of the frequency scale
            temp {int} -- electron temperature

        Returns:
            float, float -- lower and upper bound of the interval
        """
        if check:
            w_p = np.sqrt(cf.I_P['NE'] * const.elementary_charge**2
                          / (const.m_e * const.epsilon_0))
            f = w_p * (1 + 3 * cf.K_RADAR**2 *
                       temp * const.k / (const.m_e * w_p**2))**.5 / (2 * np.pi)
            upper = f + 1e6
            return bool(upper > cf.I_P['F_MAX'])
        fr = np.copy(freq)
        sp = np.copy(spec)
        w_p = np.sqrt(cf.I_P['NE'] * const.elementary_charge**2
                      / (const.m_e * const.epsilon_0))
        f = w_p * (1 + 3 * cf.K_RADAR**2 *
                   temp * const.k / (const.m_e * w_p**2))**.5 / (2 * np.pi)
        lower, upper = (f - 1e6) / 10**scale, (f + 1e6) / 10**scale
        m = (fr > lower) & (fr < upper)
        fr_n = fr[m]
        sp = sp[m]
        av = fr_n[np.argmax(sp)]
        lower, upper = av - 2e6 / 10**scale, av + 2e6 / 10**scale
        return lower, upper

    @staticmethod
    def only_ionline(f, Is):
        idx = np.argwhere(abs(f) < 4e4)
        f = f[idx].reshape((-1,))
        if isinstance(Is, list):
            for i, _ in enumerate(Is):
                Is[i] = Is[i][idx].reshape((-1,))
        else:
            Is = Is[idx].reshape((-1,))
        return f, Is

    @staticmethod
    def scaling_y(multi_params):
        y_min, y_max = np.inf, - np.inf
        for params in multi_params:
            if isinstance(params, list):
                for s in params:
                    if y_min > np.min(s):
                        y_min = np.min(s)
                    if y_max < np.max(s):
                        y_max = np.max(s)
            else:
                if y_min > np.min(params):
                    y_min = np.min(params)
                if y_max < np.max(params):
                    y_max = np.max(params)
        return y_min, y_max

if __name__ == '__main__':
    ver = 'kappa'
    kwargs = {'kappa': [3, 20], 'plasma': True}
    PlotClass(ver,  **kwargs)
