"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import functions as func
import config as cf


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


def plot_IS_spectrum():
    args = decide_on_params()
    t0 = time.perf_counter()
    Is, w = func.isspec_ne(*args)
    print(f'RunTime = {time.perf_counter() - t0}')
    plt.figure()
    plt.title('ISR spectrum')
    plt.plot(w / (2 * np.pi * cf.N_POINTS), abs(Is), 'r')
    plt.plot(- w / (2 * np.pi * cf.N_POINTS), abs(Is), 'r')
    plt.show()


if __name__ == '__main__':
    plot_IS_spectrum()
