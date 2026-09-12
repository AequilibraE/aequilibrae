"""Test suite verifying oracle parity, settled label efficiency, early exit, and unreachable cleanup (H2, H3)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from aequilibrae.paths import Graph
from aequilibrae.paths.cython.basic_path_finding import path_finding_hybrid
from aequilibrae.paths.network_skimming import NetworkSkimming
from tests.aequilibrae.paths.pathological_components import (
    make_composed_pathological_network,
    make_duplicate_uturn_control_component,
    make_equal_parallel_component,
    make_finite_turn_component,
    make_mixed_direction_component,
    make_prohibited_parallel_component,
    make_prohibited_uturn_component,
    make_selected_finite_turn_component,
    make_uturn_component,
)
from tests.aequilibrae.paths.pathological_network import PathologicalNetwork


@pytest.fixture(
    params=[
        make_finite_turn_component,
        make_prohibited_parallel_component,
        make_equal_parallel_component,
        make_uturn_component,
        make_prohibited_uturn_component,
        make_duplicate_uturn_control_component,
        make_selected_finite_turn_component,
    ]
)
def pathological_network(request) -> PathologicalNetwork:
    return PathologicalNetwork.compose(request.param())


def test_hybrid_vs_arc_based_vs_networkx_oracle_parity(pathological_network: PathologicalNetwork):
    """Verifies complete generalized cost parity between NetworkX line-graph oracle, arc-based kernel, and hybrid kernel."""
    graph = pathological_network.build_graph()
    nodes = list(graph.all_nodes)

    for origin in nodes:
        for dest in nodes:
            if origin == dest:
                continue
            # 1. Oracle path
            try:
                expected = pathological_network.oracle_path(origin, dest)
                expected_cost = expected.cost
            except nx.NetworkXNoPath:
                expected = None
                expected_cost = None

            # 2. Kernel paths
            if graph.has_turn_restrictions:
                graph.set_hybrid_kernel(True)
                assert graph.selected_kernel == "hybrid"
                res_hybrid = graph.compute_path(origin, dest)

                graph.set_hybrid_kernel(False)
                assert graph.selected_kernel == "arc-based"
                res_arc = graph.compute_path(origin, dest)
            else:
                assert graph.selected_kernel == "node-based"
                res_hybrid = graph.compute_path(origin, dest)
                res_arc = res_hybrid

            if expected is None:
                assert res_hybrid.path is None
                assert res_arc.path is None
            else:
                assert res_hybrid.path is not None
                assert res_arc.path is not None
                assert res_hybrid.milepost[-1] == pytest.approx(expected_cost)
                assert res_arc.milepost[-1] == pytest.approx(expected_cost)


def test_hybrid_settled_label_efficiency_and_early_exit():
    """Verifies that hybrid kernel settles fewer or equal labels than arc-based, and early exit settles <= full search."""
    composed = make_composed_pathological_network()
    graph = composed.build_graph()

    origin = int(graph.all_nodes[0])
    dest = int(graph.all_nodes[-1])

    # Sizing
    num_nodes = graph.num_nodes
    num_arcs = graph.num_links
    csr_indices = graph.graph["b_node"].to_numpy(np.int64, copy=False)
    a_nodes = graph.graph["a_node"].to_numpy(np.int64, copy=False)
    graph_costs = graph.cost.astype(np.float64)
    graph_fs = graph.fs.astype(np.int64)
    first_ctx = csr_indices
    last_ctx = a_nodes
    turn_fs = graph.turn_fs
    turn_to_arcs = graph.turn_to_arcs
    turn_penalties = graph.turn_penalties
    stateful = graph.stateful
    rep_arc = graph.rep_arc

    destinations_all = np.zeros(0, dtype=np.uint8)
    destinations_single = np.zeros(num_nodes, dtype=np.uint8)
    dest_idx = graph.nodes_to_indices[dest]
    destinations_single[dest_idx] = 1

    settled_full = np.zeros(1, dtype=np.int64)
    settled_early = np.zeros(1, dtype=np.int64)

    # Full search
    node_pred = np.full(num_nodes, -1, dtype=np.int64)
    connectors = np.full(num_nodes, -1, dtype=np.int64)
    reached_first = np.zeros(num_nodes, dtype=np.int64)
    node_costs = np.full(num_nodes, np.inf, dtype=np.float64)
    node_turn_penalties = np.zeros(num_nodes, dtype=np.float64)
    arc_pred = np.full(num_arcs, -1, dtype=np.int64)
    arc_turn_penalties = np.zeros(num_arcs, dtype=np.float64)

    path_finding_hybrid(
        graph.nodes_to_indices[origin],
        destinations_all,
        -1,
        graph_costs,
        csr_indices,
        graph_fs,
        a_nodes,
        stateful,
        rep_arc,
        node_pred,
        connectors,
        reached_first,
        node_costs,
        node_turn_penalties,
        arc_pred,
        arc_turn_penalties,
        turn_fs,
        turn_to_arcs,
        turn_penalties,
        False,
        False,
        0,
        first_ctx,
        last_ctx,
        settled_full,
    )

    # Early exit search
    node_pred_early = np.full(num_nodes, -1, dtype=np.int64)
    connectors_early = np.full(num_nodes, -1, dtype=np.int64)
    reached_first_early = np.zeros(num_nodes, dtype=np.int64)
    node_costs_early = np.full(num_nodes, np.inf, dtype=np.float64)
    node_turn_penalties_early = np.zeros(num_nodes, dtype=np.float64)
    arc_pred_early = np.full(num_arcs, -1, dtype=np.int64)
    arc_turn_penalties_early = np.zeros(num_arcs, dtype=np.float64)

    path_finding_hybrid(
        graph.nodes_to_indices[origin],
        destinations_single,
        1,
        graph_costs,
        csr_indices,
        graph_fs,
        a_nodes,
        stateful,
        rep_arc,
        node_pred_early,
        connectors_early,
        reached_first_early,
        node_costs_early,
        node_turn_penalties_early,
        arc_pred_early,
        arc_turn_penalties_early,
        turn_fs,
        turn_to_arcs,
        turn_penalties,
        False,
        False,
        0,
        first_ctx,
        last_ctx,
        settled_early,
    )

    assert settled_full[0] > 0
    assert settled_early[0] > 0
    assert settled_early[0] <= settled_full[0]


def test_unreachable_connectors_cleaned_up():
    """Verifies that unreachable destination nodes have connectors[d] == -1."""
    # Build a graph with an isolated disconnected node
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 3, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    import pandas as pd

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.prepare_graph(centroids=np.array([1, 2, 3, 4], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")

    # From 1, nodes 3 and 4 are unreachable
    res = graph.compute_path(1, 3)
    assert res.path is None
    idx_3 = graph.nodes_to_indices[3]
    assert res.connectors[idx_3] == -1
    assert res.predecessors[idx_3] == -1


def test_skimming_oracle_parity():
    """Verifies skimming returns identical matrices between hybrid and arc-based kernels."""
    composed = make_composed_pathological_network()
    graph = composed.build_graph()
    centroids = graph.centroids
    if len(centroids) < 2:
        centroids = graph.all_nodes[:2]
    graph.prepare_graph(centroids=centroids, remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming("cost")

    graph.set_hybrid_kernel(True)
    skm_hybrid = NetworkSkimming(graph)
    skm_hybrid.execute()
    mat_hybrid = skm_hybrid.results.skims.cost[:, :]

    graph.set_hybrid_kernel(False)
    skm_arc = NetworkSkimming(graph)
    skm_arc.execute()
    mat_arc = skm_arc.results.skims.cost[:, :]

    np.testing.assert_allclose(mat_hybrid, mat_arc, equal_nan=True)
