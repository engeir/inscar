import unittest
import numpy as np

import tool


class TestISR(unittest.TestCase):
    def test_ISR(self):
        a, b = tool.isr_spectrum('maxwell', kappa=6)
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape, msg='a.shape != b.shape')
        a, b = tool.isr_spectrum('kappa', kappa=4)
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape, msg='a.shape != b.shape')
        a, b = tool.isr_spectrum('long_calc', vdf='kappa', kappa=6)
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape, msg='a.shape != b.shape')


if __name__ == '__main__':
    unittest.main()
