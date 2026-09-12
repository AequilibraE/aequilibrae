"""Tests verifying generation-based cache invalidation and bit-identical determinism (H8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aequilibrae.paths import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming


def _sample_graph():
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 2, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 4, "a_node": 3, "b_node": 5, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.prepare_graph(centroids=np.array([1, 3, 4, 5], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")
    return graph


def test_generation_based_cache_invalidation():
    """H8: Generation counters increment and the effective-via cache follows the change, not the old value."""
    graph = _sample_graph()
    g_gen_0 = graph._graph_generation
    tr_gen_0 = graph._turn_restrictions_generation

    # 1. Setting turn restrictions must increment _turn_restrictions_generation
    turns_1 = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.0}])
    graph.set_turn_restrictions(turns_1, allow_path_uturns=False)
    assert graph._turn_restrictions_generation > tr_gen_0
    tr_gen_1 = graph._turn_restrictions_generation

    # Populate the cache, then move the restriction to a different via node. The cached entry
    # must not survive: this is the failure the generation counters exist to prevent.
    assert graph._compute_effective_turn_vias() == {2}

    turns_2 = pd.DataFrame([{"from_node": 2, "via_node": 3, "to_node": 5, "penalty": 5.0}])
    graph.set_turn_restrictions(turns_2, allow_path_uturns=False)
    assert graph._turn_restrictions_generation > tr_gen_1
    assert graph._compute_effective_turn_vias() == {3}

    # 2. Re-preparing the topology must bump the graph generation as well
    g_gen_1 = graph._graph_generation
    assert g_gen_1 > g_gen_0
    graph.prepare_graph(centroids=np.array([1, 3, 4, 5], dtype=np.int64), remove_dead_ends=False)
    assert graph._graph_generation > g_gen_1

    # Clearing turn restrictions increments generation
    tr_gen_2 = graph._turn_restrictions_generation
    graph.clear_turn_restrictions()
    assert graph._turn_restrictions_generation > tr_gen_2
    assert len(graph._compute_effective_turn_vias()) == 0


def test_bit_identical_determinism_across_multiple_runs():
    """H8: Repeated path computation and skimming runs produce bit-identical results."""
    graph = _sample_graph()
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 1.5}])
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(centroids=np.array([1, 3, 4, 5], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming("cost")

    # Run 1
    skm1 = NetworkSkimming(graph)
    skm1.set_cores(1)
    skm1.execute()
    matrix1 = np.array(skm1.results.skims.cost[:, :], copy=True)

    # Run 2, same settings
    skm2 = NetworkSkimming(graph)
    skm2.set_cores(1)
    skm2.execute()
    matrix2 = np.array(skm2.results.skims.cost[:, :], copy=True)

    # Run 3, pooled: thread count must not change the answer either
    skm3 = NetworkSkimming(graph)
    skm3.set_cores(4)
    skm3.execute()
    matrix3 = np.array(skm3.results.skims.cost[:, :], copy=True)

    # Must be bit-identical
    np.testing.assert_array_equal(matrix1, matrix2)
    np.testing.assert_array_equal(matrix1, matrix3)

    # A freshly prepared graph must reproduce the same numbers as well.
    rebuilt = _sample_graph()
    rebuilt.set_turn_restrictions(turns, allow_path_uturns=False)
    rebuilt.set_graph("cost")
    rebuilt.set_skimming("cost")
    skm4 = NetworkSkimming(rebuilt)
    skm4.set_cores(1)
    skm4.execute()
    np.testing.assert_array_equal(matrix1, np.array(skm4.results.skims.cost[:, :], copy=True))
