"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import os
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import config as cf
import tool


def loglog(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.loglog(f, Is, 'r')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_y(f, Is):
    plt.figure()
    # plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('10*log10(Power) [dB]')
    # plt.semilogy(f, Is, 'r')
    plt.plot(f, 10 * np.log10(Is), 'r')
    # plt.yscale('log')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_x(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    # plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('Power')
    plt.semilogx(f, Is, 'r')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def two_side_lin_plot(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.plot(f, Is, 'r')
    # plt.plot(- f, Is, 'r')
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.tight_layout()


def plot_IS_spectrum(version):
    f, Is = tool.isr_spectrum(version)
    save = input(
        'Press "y/yes" to save plot, any other key to dismiss.\t').lower()
    two_side_lin_plot(f, Is)
    loglog(f, Is)
    # semilog_x(f, Is)
    semilog_y(f, Is)
    if save in ['y', 'yes']:
        I_P = dict(cf.I_P, **{'kappa': cf.KAPPA,
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
        plt.savefig(pdffig, bbox_inches='tight', format='pdf', dpi=600)
        pdffig.close()
    plt.show()


if __name__ == '__main__':
    # TODO: when both functions are run using the same version, we do not need to calculate Fe and Fi twice.
    plot_IS_spectrum('kappa')
    # tool.H_spectrum('kappa')
