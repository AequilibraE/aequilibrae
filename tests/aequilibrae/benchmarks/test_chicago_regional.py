"""Chicago Regional TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    run_validation,
)

MODEL_STUB = "ChicagoRegional"


@pytest.fixture(scope="module")
def model_stub():
    return MODEL_STUB


@pytest.fixture(scope="module")
def model_folder(tntp_root):
    return tntp_root / "chicago-regional"


@pytest.mark.parametrize("algorithm", METHODS)
def test_chicago_regional(benchmark, tntp_graph, tntp_matrix, tntp_reference, algorithm, model_stub):
    run_validation(
        benchmark,
        tntp_graph,
        tntp_matrix,
        tntp_reference,
        model_stub,
        algorithm,
    )
