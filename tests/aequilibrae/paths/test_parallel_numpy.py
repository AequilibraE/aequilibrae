import unittest
import numpy as np
from aequilibrae.paths.AoN import copy_one_dimension, sum_axis1, linear_combination, linear_combination_skims
from aequilibrae.paths.AoN import copy_two_dimensions, copy_three_dimensions


class TestParallel(unittest.TestCase):
    def test_sum_axis1(self):
        target = np.zeros(50)
        source = np.random.rand(200).reshape(50, 4)
        b = np.sum(source, axis=1)

        sum_axis1(target, source, 1)
        self.assertEqual((b - target).max(), 0, "Sum Axis 1 failed")

    def test_linear_combination(self):
        target = np.zeros((50, 1))
        source = np.random.rand(50).reshape(50, 1)

        linear_combination(target, source, target, 0.05, 1)
        self.assertEqual((source - target * 20).max(), 0, "Linear combination failed")

    def test_linear_combination_skims(self):
        target = np.zeros((5, 5, 1))
        source = np.random.rand(25).reshape(5, 5, 1)

        linear_combination_skims(target, source, target, 0.8, 1)
        self.assertEqual((source - target * 5 / 4).max(), 0, "Linear combination failed")

    def test_triple_linear_combination(self):
        pass

    def test_triple_linear_combination_skims(self):
        pass

    def test_copy_one_dimension(self):
        target = np.zeros(50)
        source = np.random.rand(50)

        copy_one_dimension(target, source, 1)

        self.assertEqual(target.sum(), source.sum(), "Copying one dimension returned different values")
        if target.sum() == 0:
            self.fail("Target and source are the other way around for copying one dimension")

    def test_copy_two_dimensions(self):
        target = np.zeros(50).reshape(10, 5)
        source = np.random.rand(50).reshape(10, 5)

        copy_two_dimensions(target, source, 1)

        self.assertEqual(target.sum(), source.sum(), "Copying one dimension returned different values")
        if target.sum() == 0:
            self.fail("Target and source are the other way around for copying one dimension")

    def test_copy_three_dimensions(self):
        target = np.zeros(50).reshape(5, 2, 5)
        source = np.random.rand(50).reshape(5, 2, 5)

        copy_three_dimensions(target, source, 1)

        self.assertEqual(target.sum(), source.sum(), "Copying one dimension returned different values")
        if target.sum() == 0:
            self.fail("Target and source are the other way around for copying one dimension")


if __name__ == "__main__":
    unittest.main()
