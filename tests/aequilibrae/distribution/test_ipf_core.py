import numpy as np
import pytest

from aequilibrae.distribution.cython.ipf_core import ipf_core


@pytest.fixture
def problem():
    rng = np.random.default_rng(42)
    seed = rng.random((120, 90))
    seed[rng.random(seed.shape) < 0.2] = 0
    productions, attractions = rng.random(120) * 100, rng.random(90) * 100
    productions[3] = attractions[5] = 0
    attractions *= productions.sum() / attractions.sum()
    return seed, productions, attractions


@pytest.mark.parametrize("cores", [1, 0])
def test_ipf_core_matches_classic_ipf(problem, cores):
    """Test that ipf_core gives the iterates of row-then-column IPF, which empties rows and columns without targets."""
    seed, productions, attractions = problem
    expected = seed.copy()
    for _ in range(5):
        for axis, targets in ((1, productions), (0, attractions)):
            totals = expected.sum(axis)
            factors = np.divide(targets, totals, out=np.zeros_like(totals), where=totals > 0)
            expected *= factors[:, None] if axis else factors

    result = seed.copy()
    assert ipf_core(result, productions, attractions, 5, -1, cores, warn=False)[0] == 5
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ipf_core_converges_to_targets(problem, dtype):
    """Test that ipf_core converges to the row and column targets for float64 and float32 seeds."""
    seed, productions, attractions = problem
    result = seed.astype(dtype)
    assert ipf_core(result, productions, attractions, tolerance=1e-6)[1] < 1e-6
    np.testing.assert_allclose(result.sum(axis=1), productions, rtol=1e-4)
    np.testing.assert_allclose(result.sum(axis=0), attractions, rtol=1e-4)
