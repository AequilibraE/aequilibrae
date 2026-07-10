from multiprocessing import cpu_count

import numpy as np
from aequilibrae.paths.cython.vdf_core import akcelik, delta_akcelik


def test_akcelik_function():
    cores = cpu_count()

    num_links = 11
    congested_times = np.zeros(num_links)
    capacity = np.ones(num_links)
    fftime = np.ones(num_links)
    alpha = np.full(num_links, 0.25)
    tau = np.full(num_links, 8.0)
    length = np.ones(num_links)
    delta = np.zeros(num_links)

    link_flows = np.arange(num_links, dtype="float64") * 0.2

    akcelik(congested_times, link_flows, capacity, fftime, alpha, tau, length, cores)

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
    for i in range(1, num_links):
        link_flows.fill(1 * 0.2 * i)
        link_flows += np.arange(num_links) * dx

        akcelik(congested_times, link_flows, capacity, fftime, alpha, tau, length, cores)
        delta_akcelik(delta, link_flows, capacity, fftime, alpha, tau, length, cores)

        # We check if the analytical solution matches the numerical differentiation
        dydx = (congested_times[1] - congested_times[0]) / dx
        np.testing.assert_allclose(dydx, delta, err_msg="Problems with derivative for the akcelik vdf")
