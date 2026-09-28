"""SiouxFalls TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    run_validation,
)
from .path_finding_benchmark import run_path_finding_search

MODEL_STUB = "SiouxFalls"


@pytest.fixture(scope="module")
def model_stub():
    return MODEL_STUB


@pytest.fixture(scope="module")
def model_folder(tntp_root, model_stub):
    return tntp_root / model_stub


@pytest.mark.parametrize("algorithm", METHODS)
def test_sioux_falls(benchmark, tntp_graph, tntp_matrix, tntp_reference, algorithm, model_stub):
    run_validation(
        benchmark,
        tntp_graph,
        tntp_matrix,
        tntp_reference,
        model_stub,
        algorithm,
    )


@pytest.mark.parametrize("algorithm", ["dijkstra", "a_star"], ids=["dijkstra", "a_star"])
@pytest.mark.parametrize("turn_penalties", [False, True], ids=["without_turn_penalties", "with_turn_penalties"])
def test_sioux_falls_path_finding(benchmark, tntp_graph, model_stub, model_folder, algorithm, turn_penalties):
    run_path_finding_search(benchmark, tntp_graph, model_stub, model_folder, algorithm, turn_penalties)
