import pytest
from aequilibrae.paths.AoN import akcelik, delta_akcelik
from multiprocessing import cpu_count
import numpy as np


def test_akcelik_function():
    cores = cpu_count()

    alpha = np.zeros(11)
    tau = np.zeros(11)
    fftime = np.zeros(11)
    capacity = np.zeros(11)
    congested_times = np.zeros(11)
    delta = np.zeros(11)

    alpha.fill(0.25)
    tau.fill(8.0)
    fftime.fill(1)
    capacity.fill(1)
    link_flows = np.arange(11).astype(float) * 0.2

    akcelik(congested_times, link_flows, capacity, fftime, alpha, tau, cores)

    should_be = np.array(
        [
            1.0,
            1.17416574,
            1.32169906,
            1.45677644,
            1.58442888,
            1.70710678,
            1.82620873,
            1.94261498,
            2.05691786,
            2.16953597,
            2.28077641,
        ]
    )

    np.testing.assert_allclose(should_be, congested_times, err_msg="Akcelik is wrong")

    # Let's check the derivative for sections of the curve
    dx = 0.00000001
    for i in range(1, 11):
        link_flows.fill(1 * 0.2 * i)
        link_flows += np.arange(11) * dx

        akcelik(congested_times, link_flows, capacity, fftime, alpha, tau, cores)
        delta_akcelik(delta, link_flows, capacity, fftime, alpha, tau, cores)

        # We check if the analytical solution matches the numerical differentiation
        dydx = (congested_times[1] - congested_times[0]) / dx
        np.testing.assert_allclose(dydx, delta, err_msg="Problems with derivative for the akcelik vdf")
