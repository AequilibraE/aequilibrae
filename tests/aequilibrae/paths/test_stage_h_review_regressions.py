"""Regressions for defects found reviewing the Stage H implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.graph import GraphBase
from aequilibrae.paths.traffic_class import TrafficClass


def _scrambled_supernet_graph() -> Graph:
    """Graph whose __supernet_id__ is not the identity, so index-space mistakes are visible."""
    # Bidirectional links whose link_ids sort differently from their (a_node, b_node) order.
    # The two directions carry different costs, so a mis-ordered cost vector changes path
    # totals rather than cancelling out.
    links = [
        {"link_id": 5, "a_node": 1, "b_node": 2, "direction": 0, "distance": 1.0, "cost_ab": 1.0, "cost_ba": 10.0},
        {"link_id": 1, "a_node": 2, "b_node": 3, "direction": 0, "distance": 2.0, "cost_ab": 2.0, "cost_ba": 20.0},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 0, "distance": 4.0, "cost_ab": 4.0, "cost_ba": 40.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")
    return graph


def test_skim_congested_reindexes_supernet_ordered_costs():
    """Verifies skim_congested gathers supernet-indexed cost vectors into graph row order."""
    graph = _scrambled_supernet_graph()
    supernet_ids = graph.graph.__supernet_id__.to_numpy(copy=False)
    # Guard the fixture: an identity permutation would make this test vacuous.
    assert not np.array_equal(supernet_ids, np.arange(supernet_ids.shape[0]))

    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=2, matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])
    mat.matrix_view[:, :] = 1.0

    tc = TrafficClass("car", graph, mat)
    # Built the way traffic assignment builds them: indexed by __supernet_id__.
    congested_time = np.empty(graph.num_links, dtype=np.float64)
    congested_time[supernet_ids] = graph.graph["cost"].to_numpy(np.float64, copy=False)
    tc.congested_time = congested_time
    tc.fixed_cost = np.zeros(graph.num_links, dtype=np.float64)

    matrix = tc.skim_congested().results.skims
    orig = int(np.flatnonzero(matrix.index == 1)[0])
    dest = int(np.flatnonzero(matrix.index == 4)[0])
    # 1 -> 2 -> 3 -> 4 costs 1.0 + 2.0 + 4.0
    assert matrix.__assignment_cost__[orig, dest] == pytest.approx(7.0)
    assert matrix.__congested_time__[orig, dest] == pytest.approx(7.0)


def test_skim_congested_restores_skim_fields_when_skimming_raises():
    """Verifies skim_congested restores the previous skim settings even if computing skims fails."""
    graph = _scrambled_supernet_graph()
    graph.set_skimming(["distance"])
    pre_fields = list(graph.skim_fields)

    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=2, matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])
    mat.matrix_view[:, :] = 1.0

    tc = TrafficClass("car", graph, mat)
    tc.congested_time = np.zeros(graph.num_links, dtype=np.float64)
    tc.fixed_cost = np.zeros(graph.num_links, dtype=np.float64)

    with pytest.raises(ValueError):
        tc.skim_congested(skim_fields=["a_field_that_does_not_exist"])

    assert graph.skim_fields == pre_fields
    assert graph.turn_skim_fields == []


def test_empty_turn_table_keeps_global_uturn_policy():
    """Verifies an empty turn restriction table still records allow_path_uturns."""
    graph = _scrambled_supernet_graph()
    empty = pd.DataFrame(columns=["from_node", "via_node", "to_node", "penalty"])

    graph.set_turn_restrictions(empty, allow_path_uturns=True)
    assert graph._allow_path_uturns is True
    # Nothing is restricted, so the node-based kernel is still the one selected.
    assert graph.has_turn_restrictions is False
    assert graph.selected_kernel == "node-based"

    graph.set_turn_restrictions(empty, allow_path_uturns=False)
    assert graph._allow_path_uturns is False


def test_project_uturn_policy_applies_to_modes_without_restrictions(sioux_falls_example):
    """Verifies build_graphs applies the project allow_uturns setting even with no applicable turns."""
    with sioux_falls_example.db_connection as conn:
        conn.execute("UPDATE about SET infovalue='1' WHERE infoname='allow_uturns'")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM turn_restrictions").fetchone()[0] == 0

    sioux_falls_example.network.build_graphs(modes=["c"])
    graph = sioux_falls_example.network.graphs["c"]
    assert graph._allow_path_uturns is True


def test_node_turn_mapping_handles_heads_outside_the_forward_star():
    """Verifies restrictions whose head node falls outside the forward star are dropped, not fatal."""
    # Arcs 0 and 1 have heads (9, 8) beyond the forward star; arcs 2-5 are in range.
    a_by_arc = np.array([0, 9, 1, 2, 4, 5], dtype=np.int64)
    b_by_arc = np.array([9, 8, 2, 3, 5, 6], dtype=np.int64)
    fs = np.array([0, 2, 2, 4, 4, 4, 6, 6], dtype=np.int64)

    from_arcs, to_arcs, penalties = GraphBase._map_node_turn_restrictions_to_arcs(
        fs,
        a_by_arc,
        b_by_arc,
        np.array([0, 1, 4], dtype=np.int64),
        np.array([9, 2, 5], dtype=np.int64),
        np.array([8, 3, 6], dtype=np.int64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    # Only the two in-range restrictions map; the out-of-range one is skipped rather than
    # raising a broadcasting error from the single-match fast path.
    np.testing.assert_array_equal(from_arcs, np.array([2, 4], dtype=np.int64))
    np.testing.assert_array_equal(to_arcs, np.array([3, 5], dtype=np.int64))
    np.testing.assert_allclose(penalties, np.array([2.0, 3.0]))


def test_boolean_setters_accept_numpy_booleans():
    """Verifies graph configuration setters accept numpy booleans and still reject non-booleans."""
    graph = _scrambled_supernet_graph()

    # np.True_ is what Series.any(), np.all() and array comparisons return.
    graph.set_hybrid_kernel(np.False_)
    assert graph.use_hybrid is False
    graph.set_blocked_centroid_flows(np.True_)
    assert graph.block_centroid_flows is True
    graph.set_turn_restrictions(
        pd.DataFrame(columns=["from_node", "via_node", "to_node", "penalty"]), allow_path_uturns=np.True_
    )
    assert graph._allow_path_uturns is True

    other = _scrambled_supernet_graph()
    other.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=np.False_)
    assert other._remove_dead_ends is False

    with pytest.raises(TypeError, match="use_hybrid"):
        graph.set_hybrid_kernel("yes")
    with pytest.raises(TypeError, match="use_hybrid"):
        graph.set_hybrid_kernel(1)


def test_duplicate_turn_restrictions_keep_the_callers_columns():
    """Verifies collapsing duplicate movements preserves other columns and prohibition dominance."""
    graph = _scrambled_supernet_graph()
    rows = [
        {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.0, "modes": "c", "restriction_id": 10},
        {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": np.inf, "modes": "c", "restriction_id": 11},
    ]
    graph.set_turn_restrictions(pd.DataFrame(rows), allow_path_uturns=False)

    stored = graph._turn_restrictions
    # The schema must not depend on whether duplicates happened to be present.
    assert set(stored.columns) >= {"from_node", "via_node", "to_node", "penalty", "modes", "restriction_id"}
    assert len(stored) == 1
    assert np.isinf(stored.penalty.iloc[0])

    with pytest.raises(ValueError, match="Conflicting duplicate turn penalties"):
        conflicting = [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.0},
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 9.0},
        ]
        graph.set_turn_restrictions(pd.DataFrame(conflicting), allow_path_uturns=False)


def test_skim_congested_does_not_mutate_the_graph_cost_vector():
    """Verifies skimming the congested network leaves graph.cost matching the configured cost field."""
    graph = _scrambled_supernet_graph()
    before = np.array(graph.cost, copy=True)

    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=2, matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])
    mat.matrix_view[:, :] = 1.0

    tc = TrafficClass("car", graph, mat)
    tc.congested_time = np.full(graph.num_links, 7.0)
    tc.fixed_cost = np.zeros(graph.num_links, dtype=np.float64)
    tc.skim_congested()

    np.testing.assert_array_equal(graph.cost, before)
    np.testing.assert_array_equal(graph.cost, graph.graph[graph.cost_field].to_numpy(np.float64))


def test_mode_excluded_self_loops_do_not_reach_the_compact_graph():
    """Verifies mode-exclusion self-loops are dropped while genuine source self-loops survive."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0, "modes": "ct"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0, "modes": "ct"},
        # Genuine loop road at a node this mode serves.
        {"link_id": 3, "a_node": 3, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0, "modes": "ct"},
        # Network.build_graphs collapses links that do not serve the mode into self-loops.
        {"link_id": 4, "a_node": 2, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0, "modes": "t"},
    ]
    df = pd.DataFrame(links)
    df["link_type"] = "road"
    graph = Graph()
    graph.mode = "c"
    graph.cost_field = "cost"
    graph.network = df
    graph.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")

    compact_link_ids = set(graph.compact_graph.link_id.to_numpy().tolist())
    assert 3 in compact_link_ids, "a genuine source self-loop must survive contraction"
    assert 4 not in compact_link_ids, "a mode-exclusion self-loop must not reach the compact graph"
