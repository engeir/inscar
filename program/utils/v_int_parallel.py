"""Implementation of parallel computation of the velocity integrals for the integral variable y.
"""

import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm

from inputs import config as cf
from utils import integrand_functions as intf


def integrand(y, params, v, f):
    """Integrate from 0 to V_MAX with an integrand on the form e^{-iwt}f(t),
    for every value in the np.ndarray w.

    Arguments:
        y {np.ndarray} -- sample points of integration variable
        params {dict} -- plasma parameters
        v {np.ndarray} -- sample points of VDF
        f {np.ndarray} -- value of VDF at sample points

    Returns:
        np.ndarray -- the value of the velocity integral at every sample of the integration variable
    """
    idx = set(enumerate(y))
    func = partial(parallel, params, v, f)
    pool = mp.Pool()
    # tqdm give a neat progress bar for the iterative process
    with tqdm(total=len(y)) as pbar:
        for _ in pool.imap(func, idx):
            pbar.set_description("Calculating velocity integral")
            pbar.update(1)
    pool.close()
    return array


def parallel(params, v, f, index):
    array[index[0]] = intf.v_int_integrand(index[1], params, v, f)


def shared_array(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """

    shared_array_base = mp.Array(ctypes.c_double, shape[0])
    shared_arr = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_arr = shared_arr.view(np.double).reshape(*shape)
    return shared_arr


array = shared_array((int(cf.Y_N_POINTS),))
