import unittest

from aequilibrae.transit.functions.compute_line_bearing import compute_line_bearing


class TestComputeLineBearing(unittest.TestCase):
    def setUp(self) -> None:
        self.a_point = (-1.5639, -43.7397)
        self.b_point = (-1.5818, -43.7347)
        self.c_point = (-1.5865, -43.7183)
        self.d_point = (-1.5808, -43.7064)
        self.e_point = (-1.5641, -43.6992)
        self.f_point = (-1.5623, -43.7177)

    def test_compute_line_bearing(self):
        self.assertAlmostEqual(compute_line_bearing(self.a_point, self.b_point), 15.6066221)

        self.assertAlmostEqual(compute_line_bearing(self.c_point, self.d_point), 64.40597075)

        self.assertAlmostEqual(compute_line_bearing(self.e_point, self.f_point), 84.44276779)


if __name__ == "__name__":
    unittest.main()
