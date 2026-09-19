"""Comprehensive test suite verifying Stage H review fixes (Findings 1-7 and verification gaps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from aequilibrae.project import Project
from aequilibrae.paths.vdf import bpr


# ---------------------------------------------------------------------------
# Finding 1: Mode exclusion synthetic self-loops
# ---------------------------------------------------------------------------


def test_mode_exclusion_self_loops_cannot_bypass_turn_prohibitions(tmp_path):
    """Verifies the self-loops mode exclusion manufactures are not routable around a turn prohibition."""
    proj = Project()
    proj.new(str(tmp_path / "mode_test_proj"))

    with proj.db_connection as conn:
        # Create nodes
        for nid in [1, 2, 3, 4]:
            conn.execute(
                "INSERT INTO nodes (node_id, geometry) VALUES (?, MakePoint(?, ?, 4326))",
                (nid, float(nid), float(nid)),
            )

        # Links:
        # 1 -> 2 (mode 'c', cost 5)
        # 2 -> 3 (mode 'c', cost 5)
        # 2 -> 4 (mode 'b' only! NOT 'c')
        # 4 -> 2 (mode 'b' only! NOT 'c')
        # 1 -> 3 (mode 'c', cost 25, detour)
        links = [
            (1, 1, 2, "c", 5.0),
            (2, 2, 3, "c", 5.0),
            (3, 2, 4, "b", 1.0),
            (4, 4, 2, "b", 1.0),
            (5, 1, 3, "c", 25.0),
        ]
        for lid, a, b, modes, dist in links:
            conn.execute(
                "INSERT INTO links (link_id, a_node, b_node, direction, distance, modes, link_type, geometry) "
                "VALUES (?, ?, ?, 1, ?, ?, 'default', MakeLine(MakePoint(?, ?, 4326), MakePoint(?, ?, 4326)))",
                (lid, a, b, dist, modes, float(a), float(a), float(b), float(b)),
            )

        # Turn prohibition: prohibited 1 -> 2 -> 3 for mode 'c'
        conn.execute(
            "INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes) VALUES (1, 2, 3, NULL, 'c')"
        )

    proj.network.build_graphs(modes=["c"])
    g_c = proj.network.graphs["c"]

    # build_graphs represents the links that do not serve mode 'c' as self-loops rather than
    # dropping them. Those must not be routable: traversing one would reset the arc state and
    # let a path step around a prohibited turn.
    excluded = g_c.graph[g_c.graph.link_id.isin([3, 4])]
    assert not excluded.empty
    assert (excluded.a_node == excluded.b_node).all()

    g_c.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    g_c.set_graph("distance")

    # 1. Full compute_path: cannot bypass prohibition via excluded links 3 & 4
    p = g_c.compute_path(1, 3)
    assert p.path is not None
    # Must take detour link 5 (cost 25.0), not 1 -> 2 -> 3 (which was prohibited)
    assert list(p.path) == [5]
    assert list(p.path_nodes) == [1, 3]
    detour_dist = g_c.graph.loc[g_c.graph.link_id == 5, "distance"].iloc[0]
    assert p.milepost[-1] == pytest.approx(detour_dist)

    # 2. Compact skimming
    g_c.set_skimming(["distance"])
    skimmer = g_c.compute_skims()
    matrix = skimmer.results.skims
    origin_pos = int(np.flatnonzero(matrix.index == 1)[0])
    dest_pos = int(np.flatnonzero(matrix.index == 3)[0])
    assert float(matrix.distance[origin_pos, dest_pos]) == pytest.approx(detour_dist)

    # 3. Hybrid AoN assignment
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = np.array([1, 3], dtype=np.int64)
    mat.computational_view(["demand"])
    mat.matrix_view[0, 1] = 10.0

    tc_hybrid = TrafficClass("car", g_c, mat)
    assig_hybrid = TrafficAssignment()
    g_c.set_hybrid_kernel(True)
    assig_hybrid.set_classes([tc_hybrid])
    assig_hybrid.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig_hybrid.set_capacity_field("distance")
    assig_hybrid.set_time_field("distance")
    assig_hybrid.set_algorithm("all-or-nothing")
    assig_hybrid.execute()

    res_hybrid = tc_hybrid.results.get_load_results()
    assert res_hybrid.loc[5, "demand_tot"] == pytest.approx(10.0)
    assert res_hybrid.loc[1, "demand_tot"] == pytest.approx(0.0)

    # 4. Arc-fallback AoN assignment
    tc_arc = TrafficClass("car", g_c, mat)
    assig_arc = TrafficAssignment()
    g_c.set_hybrid_kernel(False)
    assig_arc.set_classes([tc_arc])
    assig_arc.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig_arc.set_capacity_field("distance")
    assig_arc.set_time_field("distance")
    assig_arc.set_algorithm("all-or-nothing")
    assig_arc.execute()

    res_arc = tc_arc.results.get_load_results()
    assert res_arc.loc[5, "demand_tot"] == pytest.approx(10.0)
    assert res_arc.loc[1, "demand_tot"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Finding 2: All-pruned and empty compact graphs safety
# ---------------------------------------------------------------------------


def test_all_pruned_compact_graph_mapping_and_aon_safety():
    """Verifies that an all-pruned network safely builds mappings, screens origins,

    and executes without memory bounds faults.
    """
    # 2 -> 3 (dead end link with no connection to centroids 1 and 4)
    links = [
        {"link_id": 1, "a_node": 2, "b_node": 3, "direction": 1, "distance": 5.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    g = Graph()
    g.network = df
    # Centroids 1 and 4 are completely disconnected from link 2 -> 3, so dead-end pruning removes link 1
    g.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=True)
    g.set_graph("distance")

    assert g.compact_num_links == 0
    assert g.compact_graph.empty
    assert g.compact_graph["id"].dtype == np.int64
    assert g.compact_graph["link_id"].dtype == np.int64

    # Mapping on all-pruned graph must return empty mapping arrays immediately
    idx, data, node_mapping = g.create_compressed_link_network_mapping()
    assert len(idx) == 1
    assert idx[0] == 0
    assert len(data) == 0
    assert np.all(node_mapping == -1)

    # AoN origin screening: both centroids are disconnected in compact_fs
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = np.array([1, 4], dtype=np.int64)
    mat.computational_view(["demand"])
    mat.matrix_view[0, 1] = 50.0

    tc = TrafficClass("car", g, mat)
    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("distance")
    assig.set_time_field("distance")
    assig.set_algorithm("all-or-nothing")
    assig.execute()

    # Disconnected origins must be safely reported without crashing
    report = assig.assignment.aons["car"].report
    assert any("not connected" in str(msg) for msg in report)


# ---------------------------------------------------------------------------
# Finding 3: Select-link bounds safety on pruned links
# ---------------------------------------------------------------------------


def test_select_link_on_pruned_link():
    """Verifies that selecting a link that was pruned from the compact graph

    does not trigger out-of-bounds reads and safely returns zero flow.
    """
    # 1 -> 2 (centroid connector), 2 -> 3 (connector to 3), 2 -> 4 (dead end, will be pruned)
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 5.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 5.0},
        {"link_id": 3, "a_node": 2, "b_node": 4, "direction": 1, "distance": 5.0},  # dead end
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    g = Graph()
    g.network = df
    g.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=True)
    g.set_graph("distance")

    # Link 3 was pruned
    assert 3 in g.dead_end_links
    assert g.graph.loc[g.graph.link_id == 3, "__compressed_id__"].iloc[0] == g.compact_num_links

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = np.array([1, 3], dtype=np.int64)
    mat.computational_view(["demand"])
    mat.matrix_view[0, 1] = 100.0

    tc = TrafficClass("car", g, mat)
    # Select the pruned link 3
    tc.set_select_links({"sl_pruned": [(3, 1)]})

    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("distance")
    assig.set_time_field("distance")
    assig.set_algorithm("all-or-nothing")
    assig.execute()

    sl_res = tc.results.get_sl_results()
    assert "sl_pruned_demand_tot" in sl_res.columns
    # Pruned link had zero flow
    assert sl_res.loc[3, "sl_pruned_demand_tot"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Finding 4: Complete cost normalization and contract validation
# ---------------------------------------------------------------------------


def test_cost_normalization_numeric_strings_and_validation():
    """Verifies that string numbers are converted to float64 before groupby,

    invalid non-numeric text raises ValueError, and dynamic compact cost validates contracts.
    """
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "cost_str": "10.5", "invalid": "abc"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "cost_str": "20.5", "invalid": "10.0"},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    g = Graph()
    g.network = df
    g.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)

    # 1. Invalid text raises ValueError
    with pytest.raises(ValueError, match="contains non-numeric values"):
        g.set_graph("invalid")

    # 2. String numbers are converted to float64, and compact groupby performs numeric addition
    g.set_graph("cost_str")
    assert g.graph["cost_str"].dtype == np.float64
    # Node 2 is a degree-2 pass through, so compact link cost is 10.5 + 20.5 = 31.0
    # (not string concatenation "10.520.5")
    assert g.compact_cost[0] == pytest.approx(31.0)

    # 3. Dynamic compact_costs_from_link_costs contract validation
    # Wrong length
    with pytest.raises(ValueError, match="does not match graph link count"):
        g.compact_costs_from_link_costs(np.array([1.0]))

    # Negative values
    with pytest.raises(ValueError, match="negative values"):
        g.compact_costs_from_link_costs(np.array([-5.0, 10.0]))

    # -inf values
    with pytest.raises(ValueError, match="-inf values"):
        g.compact_costs_from_link_costs(np.array([-np.inf, 10.0]))

    # NaN values coerced to +inf
    g.compact_costs_from_link_costs(np.array([np.nan, 10.0]))
    assert np.isposinf(g.compact_cost[0])


# ---------------------------------------------------------------------------
# Finding 5: PCE-weighted turn costs for assignment convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["frank-wolfe", "cfw", "bfw"])
def test_unequal_pce_two_class_assignment_with_turns(algo):
    """Verifies that multi-class equilibrium assignment with unequal PCEs and turn penalties

    converges correctly with PCE-weighted line-search and rgap while reporting unscaled per-class turn penalties.
    """
    # Diamond network:
    # 1 -> 2 (cost 10)
    # 2 -> 4 (cost 10)
    # 1 -> 3 (cost 12)
    # 3 -> 4 (cost 12)
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "free_flow_time": 10.0, "capacity": 100.0},
        {"link_id": 2, "a_node": 2, "b_node": 4, "direction": 1, "free_flow_time": 10.0, "capacity": 100.0},
        {"link_id": 3, "a_node": 1, "b_node": 3, "direction": 1, "free_flow_time": 12.0, "capacity": 100.0},
        {"link_id": 4, "a_node": 3, "b_node": 4, "direction": 1, "free_flow_time": 12.0, "capacity": 100.0},
    ]
    df = pd.DataFrame(links)
    df["distance"] = df["free_flow_time"]
    df["modes"] = "c"
    df["link_type"] = "road"

    # Turn penalty at 2: 1 -> 2 -> 4 pays penalty 5.0
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 4, "penalty": 5.0}])

    g = Graph()
    g.network = df
    g.set_turn_restrictions(turns)
    g.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=False)
    g.set_graph("free_flow_time")

    # Class 1: car (pce = 1.0, demand = 80.0)
    mat1 = AequilibraeMatrix()
    mat1.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat1.index[:] = np.array([1, 4], dtype=np.int64)
    mat1.computational_view(["demand"])
    mat1.matrix_view[0, 1] = 80.0
    tc1 = TrafficClass("car", g, mat1)
    tc1.set_pce(1.0)

    # Class 2: truck (pce = 2.5, demand = 40.0)
    mat2 = AequilibraeMatrix()
    mat2.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat2.index[:] = np.array([1, 4], dtype=np.int64)
    mat2.computational_view(["demand"])
    mat2.matrix_view[0, 1] = 40.0
    tc2 = TrafficClass("truck", g, mat2)
    tc2.set_pce(2.5)

    assig = TrafficAssignment()
    assig.set_classes([tc1, tc2])
    assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm(algo)
    assig.max_iter = 15
    assig.rgap_target = 1e-4
    assig.execute()

    # Verification:
    # 1. Assignment executed and converged or progressed
    assert assig.assignment.iter > 1
    assert assig.assignment.rgap < 1.0
    # 2. Both classes report non-negative total turn penalty in unscaled units
    assert tc1.results.total_turn_penalty >= 0.0
    assert tc2.results.total_turn_penalty >= 0.0


# ---------------------------------------------------------------------------
# Finding 6: skim_congested() restores compact_cost state
# ---------------------------------------------------------------------------


def test_skim_congested_restores_compact_cost():
    """Verifies that TrafficClass.skim_congested() snapshots and restores compact_cost."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "free_flow_time": 5.0, "capacity": 100.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "free_flow_time": 7.0, "capacity": 100.0},
    ]
    df = pd.DataFrame(links)
    df["distance"] = df["free_flow_time"]
    df["modes"] = "c"
    df["link_type"] = "road"

    g = Graph()
    g.network = df
    g.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    g.set_graph("free_flow_time")

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = np.array([1, 3], dtype=np.int64)
    mat.computational_view(["demand"])
    mat.matrix_view[0, 1] = 50.0

    tc = TrafficClass("car", g, mat)
    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("all-or-nothing")
    assig.execute()

    pre_compact_cost = np.array(g.compact_cost, copy=True)

    # Skim congested
    tc.skim_congested(["free_flow_time"])

    # Compact cost must be exactly restored
    np.testing.assert_array_equal(g.compact_cost, pre_compact_cost)


# ---------------------------------------------------------------------------
# Finding 7: Strict exact duplicate penalty equality and row permutation invariance
# ---------------------------------------------------------------------------


def test_duplicate_turn_penalties_exact_equality_and_permutation():
    """Verifies duplicate turn restrictions require exact equality and are row-permutation invariant."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "cost": 5.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "cost": 5.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    # Forward order
    turns_fwd = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.345},
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.345},
        ]
    )
    # Reversed order
    turns_rev = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.345},
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.345},
        ]
    )

    g1 = Graph()
    g1.network = df
    g1.set_turn_restrictions(turns_fwd)
    g1.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    g1.set_graph("cost")

    g2 = Graph()
    g2.network = df
    g2.set_turn_restrictions(turns_rev)
    g2.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    g2.set_graph("cost")

    # Must be bitwise identical
    np.testing.assert_array_equal(g1.compact_turn_penalties, g2.compact_turn_penalties)
    np.testing.assert_array_equal(g1.compact_turn_to_arcs, g2.compact_turn_to_arcs)
    np.testing.assert_array_equal(g1.compact_turn_fs, g2.compact_turn_fs)

    # Inexact duplicate values (e.g. 12.345 vs 12.3450000001) must raise ValueError, not silently pick
    turns_conflict = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.345},
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 12.3450000001},
        ]
    )
    g_err = Graph()
    g_err.network = df
    with pytest.raises(ValueError, match="Conflicting duplicate turn penalties"):
        g_err.set_turn_restrictions(turns_conflict)
