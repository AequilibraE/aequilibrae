"""Regressions for assignment data kept in project-wide supernet index space."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from aequilibrae.paths.results import AssignmentResults


def _matrix_for_graph(graph: Graph) -> AequilibraeMatrix:
    matrix = AequilibraeMatrix()
    matrix.create_empty(memory_only=True, zones=graph.num_zones, matrix_names=["demand"])
    matrix.index[:] = graph.centroids
    matrix.computational_view(["demand"])
    return matrix


def test_empty_assignment_results_keep_explicit_supernet_extent():
    network = pd.DataFrame(
        columns=["link_id", "a_node", "b_node", "direction", "distance", "modes", "link_type"]
    )
    graph = Graph()
    graph.supernet_size = 7
    graph.network = network
    graph.prepare_graph(centroids=np.array([1, 2], dtype=np.int64))

    results = AssignmentResults()
    results.prepare(graph, _matrix_for_graph(graph), supernet_size=7)

    assert results.links == 7
    assert results.link_loads.shape == (7, 1)
    assert results.total_link_loads.shape == (7,)
    assert results.crosswalk.shape == (7,)


def test_filtered_graph_field_vdf_uses_valid_defaults_for_inactive_arcs(coquimbo_example):
    project = coquimbo_example
    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    assert graph.num_links < graph.supernet_size

    graph.graph["alpha"] = 0.15
    graph.graph["beta"] = 4.0
    traffic_class = TrafficClass("car", graph, _matrix_for_graph(graph))

    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})

    active = graph.graph.__supernet_id__.to_numpy(copy=False)
    inactive = np.setdiff1d(np.arange(graph.supernet_size), active)
    assert inactive.size > 0
    np.testing.assert_allclose(assignment.vdf_parameters[0][active], 0.15)
    np.testing.assert_allclose(assignment.vdf_parameters[1][active], 4.0)
    np.testing.assert_allclose(assignment.vdf_parameters[0][inactive], 0.0)
    np.testing.assert_allclose(assignment.vdf_parameters[1][inactive], 1.0)


def test_filtered_graph_compact_costs_accept_global_assignment_vector(coquimbo_example):
    project = coquimbo_example
    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    graph.set_graph("distance")
    assert graph.num_links < graph.supernet_size

    traffic_class = TrafficClass("car", graph, _matrix_for_graph(graph))
    traffic_class.fixed_cost = np.zeros(graph.supernet_size)
    traffic_class.congested_time = np.ones(graph.supernet_size)

    skims = traffic_class.skim_congested("distance").results.skims
    assert skims.names == ["distance", "__assignment_cost__", "__congested_time__"]


def test_add_preload_rejects_duplicate_keys_without_mutating_input(sioux_falls_example):
    sioux_falls_example.network.build_graphs(modes=["c"])
    graph = sioux_falls_example.network.graphs["c"]
    assignment = TrafficAssignment()
    assignment.set_classes([TrafficClass("car", graph, _matrix_for_graph(graph))])

    arc = graph.graph.iloc[0]
    preload = pd.DataFrame(
        {
            "link_id": [arc.link_id, arc.link_id],
            "direction": [arc.direction, arc.direction],
            "preload": [1.0, 2.0],
        }
    )
    original = preload.copy(deep=True)

    with pytest.raises(ValueError, match="duplicate \\(link_id, direction\\) keys"):
        assignment.add_preload(preload, name="background")

    pd.testing.assert_frame_equal(preload, original)
    assert assignment.preloads is None


def test_add_preload_does_not_mutate_input_and_reports_duplicate_name(sioux_falls_example):
    sioux_falls_example.network.build_graphs(modes=["c"])
    graph = sioux_falls_example.network.graphs["c"]
    assignment = TrafficAssignment()
    assignment.set_classes([TrafficClass("car", graph, _matrix_for_graph(graph))])

    arc = graph.graph.iloc[0]
    preload = pd.DataFrame(
        {"link_id": [arc.link_id], "direction": [arc.direction], "preload": [3.0]}
    )
    original = preload.copy(deep=True)
    assignment.add_preload(preload, name="background")

    pd.testing.assert_frame_equal(preload, original)
    with pytest.raises(ValueError, match="duplicate name"):
        assignment.add_preload(preload, name="background")
