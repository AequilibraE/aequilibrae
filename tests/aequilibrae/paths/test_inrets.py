# ---------------------------------------------------------------------------------------------------------------------
# Portions of this file were contributed by Arthur Evrard and are
# retained under the license below: the MIT License (with added clause) under which it was
# contributed to AequilibraE. See LICENSE.TXT.
#
# Copyright (c) 2021 Arthur Evrard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute and/or sublicense
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Additional clause:
#
# Reference to the software has to be made in all documentation for
# work developed with the software.
# ---------------------------------------------------------------------------------------------------------------------

from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads

import numpy as np
from aequilibrae.paths.cython.vdf_core import inrets, delta_inrets


def test_inrets_function():
    cores = omp_get_max_threads()

    alpha = np.zeros(11)
    fftime = np.ones(11)
    capacity = np.ones(11)
    congested_times = np.zeros(11)
    delta = np.zeros(11)

    alpha.fill(0.95)
    link_flows = np.arange(11).astype(float) * 0.2

    inrets(congested_times, link_flows, fftime, capacity, cores, alpha)

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
        assert abs(should_be[i] - congested_times[i]) < 0.00001, "Inrets is wrong"

    # Let's check the derivative for sections of the curve
    dx = 0.00000001
    for i in range(1, 20):
        link_flows.fill(1 * 0.1001 * i)

        link_flows += np.arange(11) * dx
        inrets(congested_times, link_flows, fftime, capacity, cores, alpha)
        delta_inrets(delta, link_flows, fftime, capacity, cores, alpha)

        # The derivative needs to be monotonically increasing.
        assert min(delta[1:] - delta[:-1]) > 0, "Delta is not increasing as it should"

        # We check if the analytical solution matches the numerical differentiation
        for j in range(10):
            dydx = (congested_times[j + 1] - congested_times[j]) / dx
            assert abs(dydx - delta[j + 1]) < 0.000001, "Problems with derivative for the inrets vdf"
