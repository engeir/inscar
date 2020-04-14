import unittest
import numpy as np
import scipy.integrate as si

from utils import tool
from utils import vdfs


class TestISR(unittest.TestCase):

    @classmethod
    def SetUpClass(cls):
        cls.a, cls.b = None, None

    def tearDown(self):
        self.assertIsInstance(self.a, np.ndarray)
        self.assertIsInstance(self.b, np.ndarray)
        self.assertEqual(self.a.shape, self.b.shape, msg='a.shape != b.shape')

    def test_isr_maxwell(self):
        self.a, self.b = tool.isr_spectrum('maxwell', kappa=6)

    def test_isr_kappa(self):
        self.a, self.b = tool.isr_spectrum('kappa', kappa=4)

    def test_isr_long_calc(self):
        self.a, self.b = tool.isr_spectrum('long_calc', vdf='kappa', kappa=6)


class TestVDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.v = np.linspace(0, (1e7)**(1 / 3), int(1e6))**3
        cls.params = {'m': 9.1093837015e-31, 'T': 1000, 'kappa': 3}
        cls.f = None

    def tearDown(self):
        f = self.f * self.v**2 * 4 * np.pi
        res = si.simps(f, self.v)
        self.assertAlmostEqual(res, 1, places=4)

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
