"""Tests verifying conservative chain compression under turn restrictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph


def test_effective_via_node_protected_from_chain_compression():
    """Via nodes of active turn restrictions must never be compressed away into interior chain nodes."""
    # Chain: 1 -> 2 -> 3 -> 4
    # With turn restriction at 2: 1 -> 2 -> 3
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 5.0}])
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")

    # Node 2 must remain an endpoint in the compact graph because it is a via node!
    # Compact graph should NOT merge (1->2) and (2->3) into a single arc spanning across 2!
    compact_a = set(graph.compact_graph["a_node"].values)
    compact_b = set(graph.compact_graph["b_node"].values)

    idx_2 = graph.compact_nodes_to_indices[2]
    assert idx_2 in compact_a or idx_2 in compact_b

    # And computing path 1 -> 4 should include the turn penalty at 2:
    # 1.0 (link 1) + 5.0 (penalty) + 1.0 (link 2) + 1.0 (link 3) = 8.0
    res = graph.compute_path(1, 4)
    assert res.path is not None
    assert res.milepost[-1] == pytest.approx(8.0)


def test_cycle_preservation_under_chain_compression():
    """Multi-link cycles must not be eliminated when simple self-loops are dropped."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 3, "b_node": 5, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 4, "a_node": 5, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 5, "a_node": 2, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    # Prohibit direct 1 -> 2 -> 4, forcing loop 1 -> 2 -> 3 -> 5 -> 2 -> 4
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 4, "penalty": None}])
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")

    res = graph.compute_path(1, 4)
    assert res.path is not None
    assert list(res.path_nodes) == [1, 2, 3, 5, 2, 4]
    assert res.milepost[-1] == pytest.approx(5.0)
