"""Winnipeg TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    run_validation,
)
from .path_finding_benchmark import run_path_finding_search
from .skimming_benchmark import SKIMMING_CASES, run_skimming_benchmark

MODEL_STUB = "Winnipeg"


@pytest.fixture(scope="module")
def model_stub():
    return MODEL_STUB


@pytest.fixture(scope="module")
def model_folder(tntp_root, model_stub):
    return tntp_root / model_stub


@pytest.mark.parametrize("algorithm", METHODS)
def test_winnipeg(benchmark, tntp_graph, tntp_matrix, tntp_reference, algorithm, model_stub):
    run_validation(
        benchmark,
        tntp_graph,
        tntp_matrix,
        tntp_reference,
        model_stub,
        algorithm,
    )


@pytest.mark.parametrize("algorithm", ["dijkstra", "a_star"], ids=["dijkstra", "a_star"])
def test_winnipeg_path_finding(benchmark, tntp_graph, model_stub, model_folder, algorithm):
    run_path_finding_search(benchmark, tntp_graph, model_stub, model_folder, algorithm)


@pytest.mark.parametrize("routing_case", SKIMMING_CASES)
def test_winnipeg_skimming(benchmark, tntp_graph, model_stub, model_folder, routing_case):
    run_skimming_benchmark(benchmark, tntp_graph, model_stub, model_folder, routing_case)
