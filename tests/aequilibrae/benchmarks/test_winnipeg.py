"""Winnipeg TNTP validation."""

import pytest

from .conftest import (
    # HEAPS,
    METHODS,
    run_validation,
)

MODEL_STUB = "Winnipeg"


@pytest.fixture(scope="module")
def model_stub():
    return MODEL_STUB


@pytest.fixture(scope="module")
def model_folder(tntp_root, model_stub):
    return tntp_root / model_stub


# @pytest.mark.parametrize("heap", HEAPS)
@pytest.mark.parametrize("algorithm", METHODS)
def test_winnipeg(benchmark, tntp_graph, tntp_matrix, tntp_reference, algorithm, model_stub, heap):
    run_validation(
        benchmark,
        tntp_graph,
        tntp_matrix,
        tntp_reference,
        model_stub,
        algorithm,
        heap,
    )
