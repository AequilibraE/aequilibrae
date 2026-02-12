import random
import sqlite3
import string
from os.path import join, isfile
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph
from aequilibrae.paths.vdf import all_vdf_functions
from ...data import siouxfalls_project


@pytest.fixture(scope="function")
def project(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    return sioux_falls_example


@pytest.fixture(scope="function")
def car_graph(project):
    graph: Graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)
    return graph


@pytest.fixture(scope="function")
def matrix(project):
    mat = project.matrices.get_matrix("demand_omx")
    mat.computational_view()
    return mat


@pytest.fixture(scope="function")
def assigclass(car_graph, matrix):
    return TrafficClass("car", car_graph, matrix)


@pytest.fixture(scope="function")
def assignment(project):
    return TrafficAssignment(project)


def test_skim_after(project, assigclass):
    assig = TrafficAssignment(project)

    assig.add_class(assigclass)
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})

    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")

    assig.max_iter = 10
    assig.set_algorithm("msa")
    assig.execute()

    b = assig.skim_congested(["distance"], return_matrices=True)

    assert b["car"].names == ["distance", "__assignment_cost__", "__congested_time__"]

    b = assig.skim_congested(return_matrices=True)

    assert b["car"].names == ["__assignment_cost__", "__congested_time__"]


# tests/aequilibrae/paths/test_traffic_assignment.py

ALGORITHMS = ["msa", "cfw", "bfw", "frank-wolfe"]


def test_matrix_with_wrong_type(matrix, car_graph):
    matrix.matrix_view = np.array(matrix.matrix_view, np.int32)
    with pytest.raises(TypeError):
        TrafficClass("car", car_graph, matrix)


def test_set_vdf(assignment):
    with pytest.raises(ValueError):
        assignment.set_vdf("CQS")
    assignment.set_vdf("BPR")


def test_set_classes(assignment, assigclass):
    with pytest.raises(AttributeError):
        assignment.set_classes([1, 2])
    with pytest.raises(TypeError):
        assignment.set_classes(assigclass)
    assignment.set_classes([assigclass])


def test_algorithms_available(assignment):
    algs = assignment.algorithms_available()
    real = ["all-or-nothing", "msa", "frank-wolfe", "bfw", "cfw", "fw"]
    diff = [x for x in real if x not in algs]
    diff2 = [x for x in algs if x not in real]
    assert len(diff) + len(diff2) <= 0, "list of algorithms raised is wrong"


def test_set_cores(assignment, assigclass):
    with pytest.raises(Exception):
        assignment.set_cores(3)
    assignment.add_class(assigclass)
    with pytest.raises(ValueError):
        assignment.set_cores("q")
    assignment.set_cores(3)


def test_set_algorithm(assignment, assigclass):
    with pytest.raises(AttributeError):
        assignment.set_algorithm("not an algo")
    assignment.add_class(assigclass)
    with pytest.raises(Exception):
        assignment.set_algorithm("msa")
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 10
    for algo in ALGORITHMS:
        for _ in range(10):
            pass  # Placeholder for any repeated logic if needed
    with pytest.raises(AttributeError):
        assignment.set_algorithm("not a valid algorithm")


@pytest.mark.parametrize(
    "vdf,parameters",
    [
        *[(k, {"alpha": "b", "beta": "power"}) for k in all_vdf_functions if k != "akcelik"],
        ("akcelik", {"alpha": "b", "tau": "power"}),
        *[(k, {"alpha": 0.15, "beta": 4.0}) for k in all_vdf_functions if k != "akcelik"],
        ("akcelik", {"alpha": 0.25, "tau": 0.1 * 8.0}),
        ("akcelik", {"tau": 0.1 * 8.0}),
    ],
)
def test_set_vdf_parameters(assignment, assigclass, vdf, parameters):
    with pytest.raises(RuntimeError):
        assignment.set_vdf_parameters(parameters)
    assignment.set_vdf(vdf)
    assignment.add_class(assigclass)
    assignment.set_vdf_parameters(parameters)


def test_set_time_field(assignment, assigclass):
    with pytest.raises(ValueError):
        assignment.set_time_field("capacity")
    assignment.add_class(assigclass)
    N = random.randint(1, 50)
    val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    with pytest.raises(ValueError):
        assignment.set_time_field(val)
    assignment.set_time_field("free_flow_time")
    assert assignment.time_field == "free_flow_time"


def test_set_capacity_field(assignment, assigclass):
    with pytest.raises(ValueError):
        assignment.set_capacity_field("capacity")
    assignment.add_class(assigclass)
    N = random.randint(1, 50)
    val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    with pytest.raises(ValueError):
        assignment.set_capacity_field(val)
    assignment.set_capacity_field("capacity")
    assert assignment.capacity_field == "capacity"


def test_info(assignment, assigclass):
    iterations = random.randint(1, 10000)
    rgap = random.random() / 10000
    algo = choice(ALGORITHMS)
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = iterations
    assignment.rgap_target = rgap
    assignment.set_algorithm(algo)
    for _ in range(10):
        algo = "".join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])
    dct = assignment.info()
    if algo.lower() == "fw":
        algo = "frank-wolfe"
    assert dct["Algorithm"] == algo.lower(), "Algorithm not correct in info method"
    assert dct["Maximum iterations"] == iterations, "maximum iterations not correct in info method"
