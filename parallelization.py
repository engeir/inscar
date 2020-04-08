import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.special as sps
import scipy.constants as const
from tqdm import tqdm

import config as cf
import integrand_functions as intf
import tool


def integrate(w_c, m, T, Lambda_s, T_MAX, function, kappa=None):
    """Integrate from 0 to T_MAX with an integrand on the form e^{-iwt}f(t),
    for every value in the np.ndarray w.

    Arguments:
        w_c {float} -- gyro frequency [Hz]
        m {float} -- mass [kg]
        T {float} -- temperature [K]
        Lambda_s {float} -- ratio of collision frequency to gyro frequency [1]
        T_MAX {float} -- upper integration limit
        function {function} -- a python function / method (def)

    Returns:
        np.ndarray -- a scaled version of the result from the integration based on Hagfors [1968]
    """
    idx = [x for x in enumerate(cf.w)]
    func = partial(parallel, T_MAX)
    pool = mp.Pool()
    # tqdm give a neat progress bar for the iterative process
    with tqdm(total=len(cf.w)) as pbar:
        for _ in pool.imap(func, idx):
            pbar.set_description("Calculating spectrum")
            pbar.update(1)
    if function == intf.kappa_gordeyev:
        a = array / (2**(kappa - 1 / 2) * sps.gamma(kappa + 1 / 2))
    elif function == intf.long_calc:
        a = 4 * np.pi * T * const.k * array / m
    else:
        a = array
    if function == intf.long_calc:
        F = a
    else:
        F = 1 - (1j * cf.w + Lambda_s * w_c) * a
    return F


def parallel(T_MAX, index):
    array[index[0]] = tool.simpson(index[1], T_MAX)


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
