"""This script reads from folder `arecibo` and combines the
calculated electron distribution from file with a Maxwellian.
"""

import ast
import os
import sys

import numpy as np
import scipy.constants as const
from scipy.io import loadmat
import isr_spectrum as isr

import examples


def f_0_maxwell(particle: examples.RealDataParticle):
    # NOTE: Normalized to 1D
    v = particle.velocity_axis
    temp = particle.temperature
    mass = particle.mass
    A = (2 * np.pi * temp * const.k / mass) ** (-1 / 2)
    return A * np.exp(-(v ** 2) / (2 * temp * const.k / mass))


def interpolate_data(particle: examples.RealDataParticle):
    """Interpolate calculated distribution down to zero
    energy and add to a 1D Maxwellian.

    Args:
        v {np.ndarray} -- 1D velocity array
        params {dict} -- dictionary of all needed parameters

    Returns:
        np.ndarray -- 1D array of the distribution
    """
    v = particle.velocity_axis
    if os.path.basename(os.path.realpath(sys.argv[0])) != "main.py":
        path = "data/arecibo/"
        if not os.path.exists(path):
            path = "src/isr_spectrum/data/arecibo/"
        x = loadmat(path + particle.mat_file)
        data = x["fe_zmuE"]
        if isinstance(particle.pitch_angle, list):
            if all(isinstance(x, int) for x in particle.pitch_angle):
                sum_over_pitch = data[:, particle.pitch_angle, :]
                norm = len(particle.pitch_angle)
            else:
                raise ValueError("pitch_angle must be a list of integers")
        else:
            norm = 18
        sum_over_pitch = (
            np.einsum("ijk->ik", data) / norm
        )  # removes j-dimansion through dot-product
        idx = int(np.argwhere(read_dat_file("z4fe.dat") == particle.z))
        f_1 = sum_over_pitch[idx, :]
        energies = read_dat_file("E4fe.dat")
    else:
        path = "data/arecibo/"
        if not os.path.exists(path):
            path = "src/isr_spectrum/data/arecibo/"
        x = loadmat(path + particle.mat_file)
        data = x["fe_zmuE"]
        if isinstance(particle.pitch_angle, list):
            if all(isinstance(x, int) for x in particle.pitch_angle):
                sum_over_pitch = data[:, particle.pitch_angle, :]
                norm = len(particle.pitch_angle)
            else:
                raise ValueError("pitch_angle must be a list of integers")
        else:
            norm = 18
        sum_over_pitch = (
            np.einsum("ijk->ik", data) / norm
        )  # removes j-dimansion through dot-product
        idx = int(np.argwhere(read_dat_file("z4fe.dat") == particle.z))
        f_1 = sum_over_pitch[idx, :]
        energies = read_dat_file("E4fe.dat")

    velocities = (2 * energies * const.eV / particle.mass) ** 0.5
    new_f1 = np.interp(v, velocities, f_1)
    f_0 = f_0_maxwell(particle)
    f0_f1 = f_0 + new_f1

    return f0_f1


def read_dat_file(file):
    """Return the contents of a `.dat` file as a single numpy row vector.

    Arguments:
        file {str} -- the file name of the .dat file

    Returns:
        np.ndarray -- contents of the .dat file
    """
    l = np.array([])
    path = "data/arecibo/"
    if not os.path.exists(path):
        path = "src/isr_spectrum/data/arecibo/"
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
