"""Chicago Regional TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    run_validation,
)
from .path_finding_benchmark import run_path_finding_search
from .skimming_benchmark import SKIMMING_CASES_WITH_LEFT_TURNS, run_skimming_benchmark

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


@pytest.mark.parametrize("algorithm", ["dijkstra", "a_star"], ids=["dijkstra", "a_star"])
@pytest.mark.parametrize("turn_penalties", [False, True], ids=["without_turn_penalties", "with_turn_penalties"])
def test_chicago_regional_path_finding(benchmark, tntp_graph, model_stub, model_folder, algorithm, turn_penalties):
    run_path_finding_search(benchmark, tntp_graph, model_stub, model_folder, algorithm, turn_penalties)


@pytest.mark.parametrize("routing_case", SKIMMING_CASES_WITH_LEFT_TURNS)
def test_chicago_regional_skimming(benchmark, tntp_graph, model_stub, model_folder, routing_case):
    run_skimming_benchmark(benchmark, tntp_graph, model_stub, model_folder, routing_case)
