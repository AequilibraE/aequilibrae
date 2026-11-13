"""Chicago Regional TNTP validation."""

import pytest

from .conftest import (
    METHODS,
    run_validation,
)
from .path_finding_benchmark import run_path_finding_search
from .route_choice_benchmark import ROUTE_CHOICE_ROUTING_CASES, run_route_choice_benchmark
from .skimming_benchmark import SKIMMING_CASES_WITH_LEFT_TURNS, run_skimming_benchmark

MODEL_STUB = "ChicagoRegional"


@pytest.fixture(scope="module")
def model_stub():
    return MODEL_STUB


@pytest.fixture(scope="module")
def model_folder(tntp_root):
    return tntp_root / "chicago-regional"


@pytest.mark.parametrize("algorithm", METHODS)
def test_chicago_regional(
    benchmark, tntp_network, tntp_matrix, tntp_reference, algorithm, model_stub, tntp_block_centroid_flows
):
    run_validation(
        benchmark,
        tntp_network,
        tntp_matrix,
        tntp_reference,
        model_stub,
        algorithm,
        block_centroid_flows=tntp_block_centroid_flows,
    )


@pytest.mark.parametrize("algorithm", ["dijkstra", "a_star"], ids=["dijkstra", "a_star"])
@pytest.mark.parametrize("turn_penalties", [False, True], ids=["without_turn_penalties", "with_turn_penalties"])
def test_chicago_regional_path_finding(
    benchmark, tntp_network, model_stub, model_folder, algorithm, turn_penalties, tntp_block_centroid_flows
):
    run_path_finding_search(
        benchmark,
        tntp_network,
        model_stub,
        model_folder,
        algorithm,
        turn_penalties,
        block_centroid_flows=tntp_block_centroid_flows,
    )


@pytest.mark.parametrize("routing_case", SKIMMING_CASES_WITH_LEFT_TURNS)
def test_chicago_regional_skimming(
    benchmark, tntp_network, model_stub, model_folder, routing_case, tntp_block_centroid_flows
):
    run_skimming_benchmark(
        benchmark, tntp_network, model_stub, model_folder, routing_case, block_centroid_flows=tntp_block_centroid_flows
    )


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("routing_case", ROUTE_CHOICE_ROUTING_CASES)
@pytest.mark.parametrize("a_star", [False, True], ids=["dijkstra", "a_star"])
def test_chicago_regional_route_choice_generation(
    benchmark, tntp_network, model_stub, model_folder, algorithm, routing_case, a_star, tntp_block_centroid_flows
):
    run_route_choice_benchmark(
        benchmark,
        tntp_network,
        model_stub,
        model_folder,
        algorithm,
        routing_case,
        a_star,
        block_centroid_flows=tntp_block_centroid_flows,
    )


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("routing_case", ROUTE_CHOICE_ROUTING_CASES)
@pytest.mark.parametrize("a_star", [False, True], ids=["dijkstra", "a_star"])
def test_chicago_regional_route_choice_table(
    benchmark, tntp_network, model_stub, model_folder, algorithm, routing_case, a_star, tntp_block_centroid_flows
):
    run_route_choice_benchmark(
        benchmark,
        tntp_network,
        model_stub,
        model_folder,
        algorithm,
        routing_case,
        a_star,
        block_centroid_flows=tntp_block_centroid_flows,
        table_only=True,
    )
