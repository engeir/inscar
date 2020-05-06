"""This script implements test for the functions used throughout the program.

Run from directory `program` with command
python -m unittest test.test_ISR -b
"""

import multiprocessing as mp
mp.set_start_method('fork')

import unittest  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413
import scipy.integrate as si  # pylint: disable=C0413

from utils import spectrum_calculation as isr  # pylint: disable=C0413
from utils import vdfs  # pylint: disable=C0413
from inputs import config as cf  # pylint: disable=C0413


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
        I_P_copy = cf.I_P.copy()
        items = []
        for item in I_P_copy:
            if isinstance(I_P_copy[item], list):
                items.append(item)
        for item in items:
            cf.I_P[item] = I_P_copy[item][0]
        # if isinstance(cf.I_P['T_E'], list):
        #     cf.I_P['T_E'] = int(cf.I_P['T_E'][0])

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
        cls.v = np.linspace(0, (6e6)**(1 / 3), int(1e7))**3
        cls.params = {'m': 9.1093837015e-31, 'T': 1000, 'kappa': 3}
        cls.f = None

    def setUp(self):
        cf.SCALING = None

    def tearDown(self):
        # The function f is scaled with the Jacobian of cartesian to spherical
        f = self.f.f_0() * self.v**2 * 4 * np.pi
        res = si.simps(f, self.v)
        self.assertAlmostEqual(res, 1, places=3)

    def test_vdf_maxwell(self):
        self.f = vdfs.F_MAXWELL(self.v, self.params)

    def test_vdf_kappa(self):
        self.f = vdfs.F_KAPPA(self.v, self.params)

    def test_vdf_kappa_vol2(self):
        self.f = vdfs.F_KAPPA_2(self.v, self.params)

    def test_vdf_gauss_shell(self):
        self.f = vdfs.F_GAUSS_SHELL(self.v, self.params)

    def test_vdf_real_data(self):
        self.f = vdfs.F_REAL_DATA(self.v, self.params)


if __name__ == '__main__':
    unittest.main()
