"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import config as cf
import tool


def decide_on_params(default=True):
    if default == True:
        return [cf.F_ION + 1, cf.F0, cf.NE, cf.T_E, cf.NU_E, cf.MI, cf.T_I, cf.NU_I, cf.B, cf.THETA]
    while 1:
        try:
            use_default = input(
                'Do you want to use default parameters? (y/yes or n/no)\t').lower()
        except Exception:
            print('Input must be a string.')
        else:
            if use_default in ['y', 'yes']:
                return [cf.F_ION + 1, cf.F0, cf.NE, cf.T_E, cf.NU_E, cf.MI, cf.T_I, cf.NU_I, cf.B, cf.THETA]
            elif use_default in ['n', 'no']:
                text = 'Set the values of\n f_min, f_max, f_steps, f0, n_e, Te, Nu_e, mi, T_i, Nu_i, B, theta\nin that order, separated by a comma.\n '
                parameters = list(map(float, input(f'{text}').split(', ')))
                if len(parameters) == 12:
                    f_ion = np.linspace(
                        parameters[0], parameters[1], parameters[2])
                    parameters[:3] = [f_ion]
                    return parameters
            print('You must type in either of "y/yes/n/no".')


def loglog(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('log10(Power)')
    plt.loglog(f, Is, 'r')
    plt.tight_layout()


def semilog_y(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('log10(Power)')
    plt.plot(f, np.log10(Is), 'r')
    plt.tight_layout()


def semilog_x(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('Power')
    plt.plot(np.log10(f[1:]), Is[1:], 'r')
    plt.tight_layout()


def two_side_lin_plot(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.plot(f, Is, 'r')
    plt.plot(- f, Is, 'r')
    plt.tight_layout()


def plot_IS_spectrum():
    args = decide_on_params()
    f, Is = tool.isr_spectrum(*args)
    # Is, w = func.isspec_ne(*args)
    two_side_lin_plot(f, Is)
    loglog(f, Is)
    semilog_x(f, Is)
    semilog_y(f, Is)
    plt.show()


if __name__ == '__main__':
    plot_IS_spectrum()
