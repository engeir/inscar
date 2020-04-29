import os
import sys

import ast
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.constants as const


def f_0_maxwell(v, params):
    # NOTE: Normalized to 1D
    A = (2 * np.pi * params['T'] * const.k / params['m'])**(- 1 / 2)
    func = A * np.exp(- v**2 / (2 * params['T'] * const.k / params['m']))
    return func


def interpolate_data(v, params):
    # TODO: allow the .mat file to be chosen in main.py
    # TODO: choose height / z-value in main.py / config.py
    if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
        f_1 = np.linspace(1, 600, 600)
        energies = np.linspace(1, 110, 600)  # electronvolt
    else:
        if __name__ == '__main__':
            path = 'Arecibo-photo-electrons/'
        else:
            path = 'data/Arecibo-photo-electrons/'
        x = loadmat(path + 'fe_zmuE-01.mat')
        data = x['fe_zmuE']
        sum_over_pitch = np.einsum('ijk->ik', data)  # removes j-dimansion through dot-product
        count = np.argmax(sum_over_pitch, 0)
        idx = np.argmax(np.bincount(count))
        f_1 = sum_over_pitch[idx, :]
        energies = read_dat_file('E4fe.dat')

    velocities = (2 * energies * const.eV / params['m'])**.5
    new_f1 = np.interp(v, velocities, f_1, left=0, right=0)
    f_0 = f_0_maxwell(v, params)
    f0_f1 = f_0 + new_f1

    return f0_f1

def read_dat_file(file):
    """Return the contents of a .dat file as a single numpy row vector.

    Arguments:
        file {str} -- the file name of the .dat file

    Returns:
        np.ndarray -- contents of the .dat file
    """
    l = np.array([])
    if __name__ == '__main__':
        path = 'Arecibo-photo-electrons/'
    else:
        path = 'data/Arecibo-photo-electrons/'
    with open(path + file) as f:
        ll = f.readlines()
        ll = [x.strip() for x in ll]
        l = np.r_[l, ll]
    if len(l) == 1:
        for p in l:
            l = p.split()
    e = []
    for p in l:
        k = ast.literal_eval(p)
        e.append(k)
    return np.array(e)

if __name__ == '__main__':
    # theta_lims, E4fe, SzeN, timeOfDayUT, z4fe
    dat_file = read_dat_file('theta_lims.dat')
    print(dat_file)
