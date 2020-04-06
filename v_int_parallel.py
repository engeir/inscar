import ctypes
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm

import config as cf
import integrand_functions as intf


def integrand(y, params, v, f):
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
    idx = [x for x in enumerate(y)]
    func = partial(parallel, params, v, f)
    pool = mp.Pool()
    # tqdm give a neat progress bar for the iterative process
    for _ in tqdm(pool.imap(func, idx)):
        pass
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


array = shared_array((int(cf.N_POINTS),))
