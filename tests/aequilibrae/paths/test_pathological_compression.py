from itertools import pairwise, permutations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph

from .pathological_components import make_compression_component, make_dead_end_component
from .pathological_network import OraclePath, PathologicalNetwork


@pytest.fixture
def compression_network() -> PathologicalNetwork:
    """A restricted and an unrestricted degree-two branch expose unsafe contraction."""
    return PathologicalNetwork.compose(make_compression_component())


@pytest.fixture
def dead_end_network() -> PathologicalNetwork:
    """Sink-only, source-only, and two-way leaf spurs exercise each dead-end shape."""
    return PathologicalNetwork.compose(make_dead_end_component())


def _compact_endpoint_nodes(graph: Graph) -> set[int]:
    endpoints = graph.compact_graph[["a_node", "b_node"]].to_numpy(dtype=np.int64, copy=False).ravel()
    return {int(node) for node in graph.compact_all_nodes[np.unique(endpoints)]}


def _compact_skim_cost(graph: Graph, origin: int, destination: int) -> float:
    graph.set_skimming("cost")
    skimmer = graph.compute_skims(cores=1)
    matrix = skimmer.results.skims
    origin_position = int(np.flatnonzero(matrix.index == origin)[0])
    destination_position = int(np.flatnonzero(matrix.index == destination)[0])
    return float(matrix.cost[origin_position, destination_position])


def _assert_full_and_compact_costs(
    network: PathologicalNetwork,
    graph: Graph,
    expected: OraclePath,
    origin: int,
    destination: int,
) -> tuple[tuple[tuple[int, int], ...], float]:
    result = graph.compute_path(origin, destination)

    assert tuple(int(node) for node in result.path_nodes) == expected.nodes
    assert network.result_directed_links(result) == expected.directed_links
    assert network.result_generalized_cost(result) == pytest.approx(expected.cost)
    assert float(result.milepost[-1]) == pytest.approx(expected.cost)

    skim_cost = _compact_skim_cost(graph, origin, destination)
    assert skim_cost == pytest.approx(expected.cost)
    return network.result_directed_links(result), skim_cost


def _centroid_simple_path_invariant(
    network: PathologicalNetwork,
) -> tuple[set[int], set[tuple[int, int]]]:
    graph = network.networkx_graph()
    participating_nodes: set[int] = set()
    participating_arcs: set[tuple[int, int]] = set()

    for origin, destination in permutations((int(node) for node in network.centroids), 2):
        for path in nx.all_simple_paths(graph, origin, destination):
            participating_nodes.update(int(node) for node in path)
            for tail, head in pairwise(path):
                for data in graph.get_edge_data(tail, head).values():
                    participating_arcs.add((int(data["link_id"]), int(data["direction"])))

    return participating_nodes, participating_arcs


@pytest.mark.parametrize("timing", ["before_prepare", "after_prepare"])
def test_restriction_lifecycle_preserves_compact_route(
    compression_network: PathologicalNetwork,
    timing: str,
):
    origin = compression_network.node_id("compression:origin")
    via = compression_network.node_id("compression:via")
    bypass = compression_network.node_id("compression:bypass")
    destination = compression_network.node_id("compression:destination")
    expected = compression_network.oracle_path(origin, destination)
    graph = compression_network.build_graph(restriction_timing=timing)

    compact_nodes = _compact_endpoint_nodes(graph)
    assert via in compact_nodes
    assert bypass not in compact_nodes
    _assert_full_and_compact_costs(
        compression_network,
        graph,
        expected,
        origin,
        destination,
    )



def test_dead_end_removal_matches_centroid_simple_path_invariant(
    dead_end_network: PathologicalNetwork,
):
    participating_nodes, participating_arcs = _centroid_simple_path_invariant(dead_end_network)
    corridor_nodes = {
        dead_end_network.node_id("dead_end:origin"),
        dead_end_network.node_id("dead_end:main"),
        dead_end_network.node_id("dead_end:destination"),
    }
    spur_nodes = {
        dead_end_network.node_id("dead_end:sink"),
        dead_end_network.node_id("dead_end:source"),
        dead_end_network.node_id("dead_end:leaf"),
    }
    participating_links = {link_id for link_id, _ in participating_arcs}
    all_links = {arc.link_id for arc in dead_end_network.directed_arcs()}
    expected_dead_end_links = all_links - participating_links
    named_spur_links = {
        dead_end_network.link_id("dead_end:sink_only"),
        dead_end_network.link_id("dead_end:source_only"),
        dead_end_network.link_id("dead_end:bidirectional_leaf"),
    }

    assert participating_nodes == corridor_nodes
    assert expected_dead_end_links == named_spur_links

    graph = dead_end_network.build_graph(restriction_timing="none", remove_dead_ends=True)

    assert {int(link_id) for link_id in graph.dead_end_links} == expected_dead_end_links
    assert _compact_endpoint_nodes(graph).isdisjoint(spur_nodes)


def test_dead_end_removal_and_compression_preserve_od_behavior(
    dead_end_network: PathologicalNetwork,
):
    origin = dead_end_network.node_id("dead_end:origin")
    destination = dead_end_network.node_id("dead_end:destination")
    outcomes = {}

    for remove_dead_ends in (False, True):
        expected = dead_end_network.oracle_path(origin, destination)
        graph = dead_end_network.build_graph(
            restriction_timing="none",
            remove_dead_ends=remove_dead_ends,
        )
        outcomes[remove_dead_ends] = _assert_full_and_compact_costs(
            dead_end_network,
            graph,
            expected,
            origin,
            destination,
        )

    assert outcomes[False] == outcomes[True]


def test_stale_out_of_range_restriction_is_safely_non_applicable(
    dead_end_network: PathologicalNetwork,
):
    origin = dead_end_network.node_id("dead_end:origin")
    destination = dead_end_network.node_id("dead_end:destination")
    expected = dead_end_network.oracle_path(origin, destination)
    graph = dead_end_network.build_graph(restriction_timing="none", remove_dead_ends=True)
    stale_node = int(dead_end_network.node_frame().node_id.max()) + 100
    stale_restriction = pd.DataFrame(
        {
            "from_node": [origin],
            "via_node": [stale_node],
            "to_node": [destination],
            "penalty": [np.inf],
        }
    )

    graph.set_turn_restrictions(stale_restriction, allow_path_uturns=False)

    _assert_full_and_compact_costs(dead_end_network, graph, expected, origin, destination)
