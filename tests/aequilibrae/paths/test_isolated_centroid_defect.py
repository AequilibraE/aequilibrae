"""Test suite with a topology that directly exposes the undersized-array defect for trailing isolated centroids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph


def _make_minimal_network():
    # Only connects nodes 1 and 2 (a_node max = 0, b_node max = 1 after indexing)
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 10.0, "free_flow_time": 10.0},
        {"link_id": 2, "a_node": 2, "b_node": 1, "direction": 1, "distance": 10.0, "free_flow_time": 10.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    return df


@pytest.mark.parametrize("remove_dead_ends", [True, False])
@pytest.mark.parametrize("with_turns", [False, True])
def test_trailing_isolated_centroids_undersized_array_topology(remove_dead_ends: bool, with_turns: bool):
    """Topology where links only span indices 0..1 while centroids extend to index 6.

    Before the Stage H fix, in_degree and out_degree were sized from max(graph_b_nodes) + 1 (size 2)
    and max(graph_a_nodes) + 1 (size 2 or 1). Trailing centroids placed at indices 2..6 triggered
    out-of-bounds writes on centroid_idx and out-of-bounds reads/writes in _remove_dead_ends.
    """
    net = _make_minimal_network()
    # 7 centroids: 1 and 2 are connected; 10, 20, 30, 40, 50 are isolated trailing centroids
    centroids = np.array([1, 2, 10, 20, 30, 40, 50], dtype=np.int64)

    g = Graph()
    g.network = net.copy()

    if with_turns:
        # Reversal turn on 1 -> 2 -> 1
        turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 1, "penalty": 5.0}])
        g.set_turn_restrictions(turns, allow_path_uturns=True)

    # Must prepare cleanly without memory corruption or crash
    g.prepare_graph(centroids=centroids, remove_dead_ends=remove_dead_ends)
    g.set_graph("free_flow_time")

    # Invariants
    assert g.num_nodes == len(centroids)
    assert g.compact_num_nodes == len(centroids)

    # Connected pair 1 -> 2 should compute valid path
    res_connected = g.compute_path(1, 2)
    assert res_connected.path is not None
    assert res_connected.milepost[-1] == pytest.approx(10.0)

    # Isolated trailing centroid (e.g. 50) must be safely reported as unreachable without memory error
    res_isolated = g.compute_path(1, 50)
    assert res_isolated.path is None
    idx_50 = g.nodes_to_indices[50]
    assert res_isolated.predecessors[idx_50] == -1
    assert res_isolated.connectors[idx_50] == -1

    res_from_isolated = g.compute_path(50, 1)
    assert res_from_isolated.path is None


def test_one_way_link_asymmetric_degree_arrays():
    """Topology with a single one-way link 1 -> 2 where out_degree and in_degree have different sizes without fix."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "free_flow_time": 1.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    # Centroids [1, 2, 99]: a_nodes has only index 0; b_nodes has only index 1; 99 has index 2
    centroids = np.array([1, 2, 99], dtype=np.int64)

    for remove_dead_ends in [True, False]:
        g = Graph()
        g.network = df.copy()
        g.prepare_graph(centroids=centroids, remove_dead_ends=remove_dead_ends)
        g.set_graph("free_flow_time")

        assert g.num_nodes == 3
        res = g.compute_path(1, 2)
        assert res.path is not None
        assert g.compute_path(1, 99).path is None
        assert g.compute_path(99, 1).path is None
