"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import tool


def loglog(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('log10(Power)')
    plt.loglog(f, Is, 'r')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_y(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('10*log10(Power)')
    # plt.semilogy(f, Is, 'r')
    plt.plot(f, 10 * np.log10(Is), 'r')
    # plt.yscale('log')
    plt.minorticks_on()
    plt.grid(True, which="both",ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_x(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    # plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('Power')
    plt.semilogx(f, Is, 'r')
    plt.grid(True, which="both",ls="-", alpha=0.4)
    plt.tight_layout()


def two_side_lin_plot(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.plot(f, Is, 'r')
    plt.plot(- f, Is, 'r')
    plt.grid(True, which="major",ls="-", alpha=0.4)
    plt.tight_layout()


def plot_IS_spectrum():
    f, Is = tool.isr_spectrum()
    # Is, w = func.isspec_ne(*args)
    two_side_lin_plot(f, Is)
    loglog(f, Is)
    semilog_x(f, Is)
    semilog_y(f, Is)
    plt.show()


if __name__ == '__main__':
    plot_IS_spectrum()
