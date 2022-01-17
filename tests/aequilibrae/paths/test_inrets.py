from unittest import TestCase
from aequilibrae.paths.AoN import inrets, delta_inrets
from multiprocessing import cpu_count
import numpy as np


class TestInrets(TestCase):
    def test_inrets_funtion(self):
        cores = cpu_count()

        alpha = np.zeros(11)
        beta = np.zeros(11)
        fftime = np.ones(11)
        capacity = np.ones(11)
        congested_times = np.zeros(11)
        delta = np.zeros(11)

        alpha.fill(0.95)
        link_flows = np.arange(11).astype(float) * 0.2

        inrets(congested_times, link_flows, capacity, fftime, alpha, beta, cores)

        should_be = np.array(
            [
                1,
                1.011111111,
                1.028571429,
                1.06,
                1.133333333,
                1.5,
                2.16,
                2.94,
                3.84,
                4.86,
                6,
            ]
        )

        for i in range(11):
            self.assertAlmostEqual(should_be[i], congested_times[i], 5, "Inrets is wrong")

        # Let's check the derivative for sections of the curve
        dx = 0.00000001
        for i in range(1, 20):
            link_flows.fill(1 * 0.1001 * i)

            link_flows += np.arange(11) * dx
            print(link_flows)
            inrets(congested_times, link_flows, capacity, fftime, alpha, beta, cores)
            delta_inrets(delta, link_flows, capacity, fftime, alpha, beta, cores)

            # The derivative needs to be monotonically increasing.
            # np.testing.asser
            self.assertGreater(min(delta[1:] - delta[:-1]), 0, "Delta is not increasing as it should")

            # We check if the analytical solution matches the numerical differentiation
            for j in range(10):
                dydx = (congested_times[j + 1] - congested_times[j]) / dx
                self.assertAlmostEqual(dydx, delta[j + 1], 6, "Problems with derivative for the inrets vdf")
            print(j)
