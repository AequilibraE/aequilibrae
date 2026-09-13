"""Byte-for-byte determinism of compact topology, boundary context, rep-arc and turn-CSR arrays."""

from __future__ import annotations

import tempfile
import numpy as np
import pandas as pd

from aequilibrae.paths import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming


def _build_benchmark_network():
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 2.0, "free_flow_time": 2.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 3.0, "free_flow_time": 3.0},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 1, "distance": 4.0, "free_flow_time": 4.0},
        {"link_id": 4, "a_node": 1, "b_node": 5, "direction": 1, "distance": 5.0, "free_flow_time": 5.0},
        {"link_id": 5, "a_node": 5, "b_node": 4, "direction": 1, "distance": 5.0, "free_flow_time": 5.0},
        {"link_id": 6, "a_node": 2, "b_node": 5, "direction": 1, "distance": 1.0, "free_flow_time": 1.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    return df


def _build_sample_turns():
    return pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.5},
            {"from_node": 2, "via_node": 5, "to_node": 4, "penalty": 1.0},
            {"from_node": 1, "via_node": 2, "to_node": 5, "penalty": np.inf},  # prohibited
        ]
    )


def _build_and_prepare(compress: bool = True) -> Graph:
    net = _build_benchmark_network()
    turns = _build_sample_turns()
    centroids = np.array([1, 4], dtype=np.int64)

    g = Graph()
    g.network = net
    g.set_turn_restrictions(turns, allow_path_uturns=False)
    g.prepare_graph(centroids=centroids, remove_dead_ends=compress)
    g.set_graph("free_flow_time")
    return g


def _assert_graph_arrays_equal(g1: Graph, g2: Graph):
    # 1. Compact topology
    np.testing.assert_array_equal(g1.compact_all_nodes, g2.compact_all_nodes)
    np.testing.assert_array_equal(g1.compact_nodes_to_indices, g2.compact_nodes_to_indices)
    np.testing.assert_array_equal(g1.compact_fs, g2.compact_fs)
    np.testing.assert_array_equal(
        g1.compact_graph["id"].to_numpy(),
        g2.compact_graph["id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        g1.compact_graph["link_id"].to_numpy(),
        g2.compact_graph["link_id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        g1.compact_graph["a_node"].to_numpy(),
        g2.compact_graph["a_node"].to_numpy(),
    )
    np.testing.assert_array_equal(
        g1.compact_graph["b_node"].to_numpy(),
        g2.compact_graph["b_node"].to_numpy(),
    )
    np.testing.assert_array_equal(
        g1.compact_graph["direction"].to_numpy(),
        g2.compact_graph["direction"].to_numpy(),
    )
    np.testing.assert_array_equal(
        g1.graph["__compressed_id__"].to_numpy(),
        g2.graph["__compressed_id__"].to_numpy(),
    )
    if hasattr(g1, "_crosswalk") and hasattr(g2, "_crosswalk"):
        np.testing.assert_array_equal(g1._crosswalk, g2._crosswalk)

    # 2. Boundary context
    np.testing.assert_array_equal(g1._compact_first_node, g2._compact_first_node)
    np.testing.assert_array_equal(g1._compact_last_node, g2._compact_last_node)

    # 3. Representative-arc and stateful arrays
    np.testing.assert_array_equal(g1.stateful, g2.stateful)
    np.testing.assert_array_equal(g1.compact_stateful, g2.compact_stateful)
    np.testing.assert_array_equal(g1.rep_arc, g2.rep_arc)
    np.testing.assert_array_equal(g1.compact_rep_arc, g2.compact_rep_arc)

    # 4. Turn-CSR arrays
    np.testing.assert_array_equal(g1.turn_fs, g2.turn_fs)
    np.testing.assert_array_equal(g1.turn_to_arcs, g2.turn_to_arcs)
    np.testing.assert_array_equal(g1.turn_penalties, g2.turn_penalties)
    np.testing.assert_array_equal(g1.compact_turn_fs, g2.compact_turn_fs)
    np.testing.assert_array_equal(g1.compact_turn_to_arcs, g2.compact_turn_to_arcs)
    np.testing.assert_array_equal(g1.compact_turn_penalties, g2.compact_turn_penalties)


def test_determinism_across_independent_graph_builds():
    """Verifies that building a turn-restricted graph twice from scratch yields identical byte-for-byte arrays."""
    g1 = _build_and_prepare(compress=True)
    g2 = _build_and_prepare(compress=True)

    _assert_graph_arrays_equal(g1, g2)


def test_determinism_across_reprepare():
    """Verifies that re-preparing a graph in place restores bit-identical topological and turn arrays."""
    g = _build_and_prepare(compress=True)
    g_snapshot = _build_and_prepare(compress=True)

    # Trigger in-place repreparation
    g.prepare_graph(centroids=g.centroids, remove_dead_ends=True)
    g.set_graph("free_flow_time")

    _assert_graph_arrays_equal(g, g_snapshot)


def test_determinism_disk_serialization_round_trip():
    """Verifies that saving to disk and reloading preserves all CSR, topology, and boundary arrays byte-for-byte."""
    g1 = _build_and_prepare(compress=True)

    with tempfile.NamedTemporaryFile(suffix=".aeg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        g1.save_to_disk(tmp_path)
        g2 = Graph()
        g2.load_from_disk(tmp_path)

        _assert_graph_arrays_equal(g1, g2)
        assert g2.use_hybrid == g1.use_hybrid
    finally:
        import os

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_determinism_thread_invariance_in_skimming():
    """Verifies that multithreaded network skimming yields byte-identical skim results compared to 1 thread."""
    g = _build_and_prepare(compress=True)
    g.set_skimming(["free_flow_time"])

    skm_1 = NetworkSkimming(g)
    skm_1.set_cores(1)
    skm_1.execute()
    mat_1 = np.array(skm_1.results.skims.free_flow_time[:, :], copy=True)

    skm_4 = NetworkSkimming(g)
    skm_4.set_cores(4)
    skm_4.execute()
    mat_4 = np.array(skm_4.results.skims.free_flow_time[:, :], copy=True)

    np.testing.assert_array_equal(mat_1, mat_4)
