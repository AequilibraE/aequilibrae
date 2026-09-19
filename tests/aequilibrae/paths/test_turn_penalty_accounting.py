"""Tests verifying turn penalty accounting across assignment results and congested skims (H5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.traffic_class import TrafficClass


def _build_simple_turn_graph():
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.5}])
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    graph.set_graph("cost")
    return graph


def test_assignment_results_reset_clears_total_turn_penalty():
    """Verifies that AssignmentResults.reset() restores total_turn_penalty to 0.0."""
    graph = _build_simple_turn_graph()
    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=2, matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])

    res = AssignmentResults()
    res.prepare(graph, mat)
    res.total_turn_penalty = 123.45
    res.reset()
    assert res.total_turn_penalty == 0.0


def test_traffic_class_skim_congested_turn_penalties():
    """Verifies that skim_congested generalized cost includes turn penalties when assignment cost is used."""
    graph = _build_simple_turn_graph()
    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=2, matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])
    mat.matrix_view[:, :] = 1.0

    tc = TrafficClass("car", graph, mat)
    tc.congested_time = graph.graph["cost"].to_numpy(np.float64, copy=True)
    tc.fixed_cost = np.zeros(graph.num_links, dtype=np.float64)

    skimmer = tc.skim_congested()
    matrix = skimmer.results.skims

    orig_pos = int(np.flatnonzero(matrix.index == 1)[0])
    dest_pos = int(np.flatnonzero(matrix.index == 3)[0])
    # cost = link 1 (1.0) + turn (2.5) + link 2 (1.0) = 4.5
    assert matrix.__assignment_cost__[orig_pos, dest_pos] == pytest.approx(4.5)
