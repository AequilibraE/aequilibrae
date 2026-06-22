"""Barcelona TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    TNTP_ROOT,
    load_known_results,
    load_tntp_graph,
    load_tntp_matrix,
    run_validation,
)

MODEL_FOLDER = TNTP_ROOT / "Barcelona"
MODEL_STUB = "Barcelona"


@pytest.fixture(scope="module")
def tntp_matrix():
    mat = load_tntp_matrix(MODEL_FOLDER, MODEL_STUB)
    yield mat
    mat.close()


@pytest.fixture(scope="module")
def tntp_graph(tntp_matrix):
    return load_tntp_graph(MODEL_FOLDER, MODEL_STUB, tntp_matrix.index)


@pytest.fixture(scope="module")
def tntp_reference():
    return load_known_results(MODEL_FOLDER, MODEL_STUB)


@pytest.mark.benchmark
@pytest.mark.parametrize("algorithm", METHODS)
def test_barcelona(benchmark, tntp_graph, tntp_matrix, tntp_reference, algorithm):
    run_validation(benchmark, tntp_graph, tntp_matrix, tntp_reference, MODEL_STUB, algorithm)
