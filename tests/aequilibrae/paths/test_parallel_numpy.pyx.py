import unittest
import numpy as np
from aequilibrae.paths.AoN import copy_one_dimension


class Test(unittest.TestCase):
    def test_sum_axis1(self):
        self.fail()

    def test_linear_combination(self):
        self.fail()

    def test_linear_combination_skims(self):
        pass

    def test_triple_linear_combination(self):
        pass

    def test_triple_linear_combination_skims(self):
        pass

    def test_copy_one_dimension(self):
        target = np.zeros(50)
        source = np.random.rand(50)

        copy_one_dimension(target, source, 1)

        self.assertEqual(target.sum(), source.sum(), 'Copying one dimension returned different values')
        if target.sum() == 0:
            self.fail('Target and source are the other way around for copying one dimension')

    def test_copy_two_dimensions(self):
        pass

    def test_copy_three_dimensions(self):
        pass


if __name__ == '__main__':
    unittest.main()