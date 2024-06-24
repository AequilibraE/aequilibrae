import numpy as np
import pytest
from typing import List

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, AequilibraeMatrix
from aequilibrae.utils.create_example import create_example

# TODO:
# 1) Add PCE to transit database schema for each trip
# 2) Input timings surrounding midnight (ie going past 24hrs)??


@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"), from_model="coquimbo")
    proj.network.build_graphs()
    proj.activate()
    yield proj
    proj.close()


@pytest.fixture
def graph(project: Project):
    g = project.network.graphs["c"]
    g.set_skimming(["travel_time"])
    g.set_graph("travel_time")
    g.set_blocked_centroid_flows(False)
    g.graph["capacity"] = 500
    g.graph["travel_time"] = g.graph["distance"] / 50
    return g


@pytest.fixture
def demand(graph):
    n_zones = graph.centroids.shape[0]
    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=n_zones, matrix_names=["car"])
    matrix.index = graph.centroids

    matrix.matrices[:, :, 0] = 5  # 5 trips from each OD should cause a little bit of congestion
    matrix.computational_view("car")

    yield matrix

    matrix.close()


@pytest.fixture
def assignment(graph: Graph, demand):

    # Create assignment and set parameters
    assignment = TrafficAssignment()
    assignment.set_classes([TrafficClass("car", graph, demand)])

    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("travel_time")
    assignment.max_iter = 1  # AON assignment
    assignment.set_algorithm("msa")

    return assignment


def hr_to_sec(e):
    return int(e * 60 * 60)


def calc_preload(project, graph, start, end):
    return project.network.build_pt_preload([graph], hr_to_sec(start), hr_to_sec(end), inclusion_cond="start")


def test_run(project: Project, graph: Graph, assignment: TrafficAssignment):
    """Tests a full run through of pt preloading."""

    preload = calc_preload(project, graph, 7, 8)

    # Run non-preloaded assignment and get results
    assignment.execute()
    without_res = assignment.results()

    # Run preloaded assignment and get results
    assignment.set_pt_preload(preload)
    assignment.execute()
    with_res = assignment.results()

    # Check that average delay increases (ie the preload has reduced speeds)
    assert with_res["Delay_factor_AB"].mean() > without_res["Delay_factor_AB"].mean()


def test_built_pt_preload(project: Project, graph: Graph):
    """
    Check that building pt preload works correctly for a basic example from
    the coquimbo network.
    """

    preloads = [calc_preload(project, graph, start, end) for start, end in [(7, 8), (6.5, 8.5), (5, 10)]]

    # Assertions about the preload and coquimbo network:
    # Check correct size
    for preload in preloads:
        assert len(preload) == len(graph.graph)

    # Check preloads increase in size as time period increases
    for p1, p2 in zip(preloads, preloads[1:]):
        assert np.all(p1 <= p2)
        assert p1.sum() < p2.sum()
