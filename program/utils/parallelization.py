"""Implementation of parallel computation of the integrals for the frequency spectrum.
"""

import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.special as sps
import scipy.constants as const
import scipy.integrate as si
from tqdm import tqdm

from inputs import config as cf
from utils import integrand_functions as intf
from utils import spectrum_calculation as isr


def integrate(m, T, nu, y, function, kappa=None):
    """Integrate from 0 to Y_MAX with an integrand on the form e^{-iwy}f(y),
    for every value in the np.ndarray w.

    Arguments:
        m {float} -- mass [kg]
        T {float} -- temperature [K]
        nu {float} -- collision frequency [Hz]
        y {np.ndarray} -- integration sample points
        function {function} -- a python function / method (def)

    Keyword Arguments:
        kappa {int or float} -- index determining the order of the kappa VDFs (default: {None})

    Returns:
        np.ndarray -- a scaled version of the result from the integration based on Hagfors [1968]
    """
    idx = set(enumerate(cf.w))
    f = function.integrand()
    func = partial(parallel, y, f)
    pool = mp.Pool()
    # tqdm give a neat progress bar for the iterative process
    with tqdm(total=len(cf.w)) as pbar:
        for _ in pool.imap(func, idx):
            pbar.set_description("Calculating spectrum")
            pbar.update(1)
    pool.close()
    if function.type == 'kappa':
        a = array / (2**(kappa - 1 / 2) * sps.gamma(kappa + 1 / 2))
    elif function.type == 'long_calc':
        a = 4 * np.pi * T * const.k * array / m
    else:
        a = array
    if function.type == 'long_calc':
        F = a
    else:
        F = 1 - (1j * cf.w + nu) * a
    return F


def parallel(y, f, index):
    array[index[0]] = simpson(index[1], y, f)


def simpson(w, y, f):
    val = np.exp(- 1j * w * y) * f

    sint = si.simps(val, y)
    return sint


def shared_array(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """

    shared_array_base = mp.Array(ctypes.c_double, 2 * shape[0])
    shared_arr = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_arr = shared_arr.view(np.complex128).reshape(*shape)
    return shared_arr


array = shared_array((len(cf.w),))
