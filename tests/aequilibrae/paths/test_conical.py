from unittest import TestCase
from aequilibrae.paths.AoN import conical, delta_conical
from multiprocessing import cpu_count
import numpy as np


class TestConical(TestCase):
    def test_conial_funtion(self):
        cores = cpu_count()

        alpha = np.zeros(11)
        beta = np.zeros(11)
        fftime = np.zeros(11)
        capacity = np.zeros(11)
        congested_times = np.zeros(11)
        delta = np.zeros(11)

        alpha.fill(9.0)
        beta.fill(1.06)
        fftime.fill(1)
        capacity.fill(1)
        link_flows = np.arange(11).astype(float) * 0.2

        conical(congested_times, link_flows, capacity, fftime, alpha, beta, cores)

        should_be = np.array(
            [
                1,
                1.017609498,
                1.043053698,
                1.092812279,
                1.228923167,
                2,
                4.828923168,
                8.292812282,
                11.8430537,
                15.4176095,
                19.00220725,
            ]
        )

        for i in range(11):
            self.assertAlmostEqual(should_be[i], congested_times[i], 5, "Conical is wrong")

        # Let's check the derivative for sections of the curve
        dx = 0.00000001
        for i in range(1, 11):
            link_flows.fill(1 * 0.2 * i)
            link_flows += np.arange(11) * dx

            conical(congested_times, link_flows, capacity, fftime, alpha, beta, cores)
            delta_conical(delta, link_flows, capacity, fftime, alpha, beta, cores)

            # The derivative needs to be monotonically increasing.
            self.assertGreater(delta[1], delta[0], "Delta is not increasing as it should")

            # We check if the analytical solution matches the numerical differentiation
            dydx = (congested_times[1] - congested_times[0]) / dx
            self.assertAlmostEqual(dydx, delta[1], 6, "Problems with derivative for the conical vdf")
            print(dydx, delta[1])
