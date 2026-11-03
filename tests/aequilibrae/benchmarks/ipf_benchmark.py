"""IPF benchmarks adapted from docs/source/distribution_procedures/IPF_benchmark.ipynb."""

import numpy as np
from aequilibrae.distribution.cython.ipf_core import ipf_core
from threadpoolctl import threadpool_limits

ZONE_COUNTS = [1000, 2500, 5000, 10000]
CORES = [1, 4, 8]
ITERATIONS = 50
REPETITIONS = 3
RANDOM_SEED = 42


def numpy_classic(productions, attractions, trips, iterations):
    """Calculate Fratar trip distribution by rescaling rows and columns each iteration."""
    # From: https://github.com/joshchea/python-tdm/blob/master/scripts/CalcDistribution.py
    sum_productions = productions.sum()
    sum_attractions = attractions.sum()
    if sum_productions != sum_attractions:
        attractions = attractions * (sum_productions / sum_attractions)

    # Run 2D balancing.
    for _ in range(iterations):
        computed_productions = trips.sum(1)
        computed_productions[computed_productions == 0] = 1
        row_factors = productions / computed_productions
        trips = trips * row_factors[:, np.newaxis]

        computed_attractions = trips.sum(0)
        computed_attractions[computed_attractions == 0] = 1
        column_factors = attractions / computed_attractions
        trips = trips * column_factors

    return trips


def numpy_gemv(productions, attractions, trips, iterations):
    # The same factor-based algorithm, with two BLAS matrix-vector products per iteration.
    # NumPy's BLAS thread count is controlled outside the timed call.
    columns = np.ones(trips.shape[1], dtype=trips.dtype)
    productions, attractions = productions.astype(trips.dtype), attractions.astype(trips.dtype)

    for _ in range(iterations):
        rows = productions / (trips @ columns)
        columns = attractions / (rows @ trips)

    trips *= rows[:, np.newaxis]
    trips *= columns[np.newaxis, :]
    return trips


def run_ipf_benchmark(benchmark, problem, implementation, cores):
    """Time IPF alone, then compare the result with double-precision NumPy IPF."""
    seed, productions, attractions = problem

    def setup():
        # Every run gets a fresh copy of the seed, made outside the timed region.
        return (seed.copy(),), {}

    def run(trips):
        if implementation is ipf_core:
            # Use a negative tolerance to force all iterations and suppress the expected non-convergence warning.
            ipf_core(trips, productions, attractions, max_iterations=ITERATIONS, tolerance=-5, cores=cores, warn=False)
            return trips

        return implementation(productions, attractions, trips, ITERATIONS)

    with threadpool_limits(limits=cores, user_api="blas"):
        result = benchmark.pedantic(run, setup=setup, rounds=REPETITIONS, iterations=1)
        expected = numpy_gemv(productions, attractions, seed.astype(np.float64), ITERATIONS)

    rtol = 1e-5 if seed.dtype == np.float32 else 1e-12
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=0)
    np.testing.assert_allclose(result.sum(axis=1, dtype=np.float64), productions, rtol=rtol, atol=0)
    np.testing.assert_allclose(result.sum(axis=0, dtype=np.float64), attractions, rtol=rtol, atol=0)
