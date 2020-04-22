"""This script implements test for the functions used throughout the program.

Run from directory `program` with command
python -m unittest test.test_ISR -b
"""

import multiprocessing as mp
mp.set_start_method('fork')

import unittest
import numpy as np
import scipy.integrate as si
# import mpmath

from utils import spectrum_calculation as isr
from utils import vdfs
from inputs import config as cf


class TestISR(unittest.TestCase):
    """Check if the output from isr_spectrum is as expected.

    Should return two numpy.ndarrays of equal shape.

    Arguments:
        unittest {class} -- inherits from unittest to make it a TestCase
    """

    @classmethod
    def SetUpClass(cls):
        cls.a, cls.b = None, None

    def setUp(self):
        if isinstance(cf.I_P['T_E'], list):
            cf.I_P['T_E'] = int(cf.I_P['T_E'][0])

    def tearDown(self):
        self.assertIsInstance(self.a, np.ndarray)
        self.assertIsInstance(self.b, np.ndarray)
        self.assertEqual(self.a.shape, self.b.shape, msg='a.shape != b.shape')

    def test_isr_maxwell(self):
        self.a, self.b = isr.isr_spectrum('maxwell', kappa=6)

    def test_isr_kappa(self):
        self.a, self.b = isr.isr_spectrum('kappa', kappa=4)

    def test_isr_long_calc(self):
        self.a, self.b = isr.isr_spectrum('long_calc', vdf='kappa', kappa=6)


class TestVDF(unittest.TestCase):
    """Class which test if the VDFs are normalized.

    Arguments:
        unittest {class} -- inherits from unittest to make it a TestCase
    """

    @classmethod
    def setUpClass(cls):
        if isinstance(cf.I_P['T_E'], list):
            cf.I_P['T_E'] = int(cf.I_P['T_E'][0])
        cls.v = np.linspace(0, (1e8)**(1 / 3), int(1e7))**3
        cls.params = {'m': 9.1093837015e-31, 'T': 1000, 'kappa': 3}
        cls.f = None

    def tearDown(self):
        # The function f is scaled with the Jacobian of cartesian to spherical
        f = self.f * self.v**2 * 4 * np.pi
        res = si.simps(f, self.v)
        # f = lambda x: self.f(x, self.params, module=mpmath) * x**2 * 4 * mpmath.pi
        # res = mpmath.quad(f, [0, mpmath.inf])
        self.assertAlmostEqual(res, 1, places=7)

    def test_vdf_maxwell(self):
        self.f = vdfs.f_0_maxwell(self.v, self.params)

    def test_vdf_kappa(self):
        self.f = vdfs.f_0_kappa(self.v, self.params)

    def test_vdf_kappa_vol2(self):
        self.f = vdfs.f_0_kappa_two(self.v, self.params)

    def test_vdf_gauss_shell(self):
        self.f = vdfs.f_0_gauss_shell(self.v, self.params)


if __name__ == '__main__':
    unittest.main()
