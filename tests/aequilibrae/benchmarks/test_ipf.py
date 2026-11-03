"""IPF benchmarks for AequilibraE and the NumPy reference implementations."""

import numpy as np
import pytest

from aequilibrae.distribution.cython.ipf_core import ipf_core

from .ipf_benchmark import CORES, RANDOM_SEED, ZONE_COUNTS, numpy_classic, numpy_gemv, run_ipf_benchmark


@pytest.fixture(scope="module", params=ZONE_COUNTS)
def zones(request):
    return request.param


@pytest.fixture(scope="module", params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def ipf_problem(zones, dtype):
    rng = np.random.default_rng(RANDOM_SEED)
    seed = rng.random((zones, zones)).astype(dtype, copy=False)
    productions = rng.random(zones) * 1000
    attractions = rng.random(zones) * 1000
    attractions *= productions.sum() / attractions.sum()
    return seed, productions, attractions


@pytest.mark.parametrize(
    "implementation", [ipf_core, numpy_classic, numpy_gemv], ids=["ipf_core", "numpy_classic", "numpy_gemv"]
)
@pytest.mark.parametrize("cores", CORES)
def test_ipf(benchmark, ipf_problem, implementation, cores):
    run_ipf_benchmark(benchmark, ipf_problem, implementation, cores)
