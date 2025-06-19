import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths import Graph
from os.path import join, dirname
from uuid import uuid4
from shutil import copytree, rmtree
from aequilibrae.project import Project
from aequilibrae.paths.results import PathResults
from aequilibrae.utils.create_example import create_example
from aequilibrae.transit import Transit
from ...data import siouxfalls_project, path_test, test_graph


@pytest.fixture
def setup_project(tmp_path):
    temp_proj_folder = tmp_path / uuid4().hex
    copytree(siouxfalls_project, temp_proj_folder)
    project = Project.from_path(temp_proj_folder)
    project.network.build_graphs()
    yield project
    project.close()
    rmtree(temp_proj_folder, ignore_errors=True)


@pytest.fixture
def setup_graph(setup_project):
    return setup_project.network.graphs["c"]

def test_upper_case_variables(setup_graph):
    network = setup_graph.network
    network.columns = network.columns.str.upper()
    g = Graph()
    g.network = network
    assert g.network.columns.tolist() == setup_graph.network.columns.tolist(), "Graph columns are not lower case"


def test_prepare_graph(setup_graph):
    setup_graph.prepare_graph(np.arange(5) + 1)


def test_prepare_graph_no_centroids(setup_graph):
    setup_graph.prepare_graph()
    setup_graph.set_graph("distance")
    setup_graph.set_skimming("distance")


def test_set_graph(setup_graph):
    setup_graph.set_graph(cost_field="distance")
    setup_graph.set_blocked_centroid_flows(block_centroid_flows=True)
    assert setup_graph.num_zones == 24, "Number of centroids not properly set"
    assert setup_graph.num_links == 76, "Number of links not properly set"
    assert setup_graph.num_nodes == 24, f"Number of nodes not properly set - {setup_graph.num_nodes}"


def test_save_to_disk(setup_graph):
    setup_graph.save_to_disk(join(path_test, "aequilibrae_test_graph.aeg"))
    graph_id = setup_graph._id
    assert graph_id is not None


def test_load_from_disk():
    reference_graph = Graph()
    reference_graph.load_from_disk(test_graph)

    new_graph = Graph()
    new_graph.load_from_disk(join(path_test, "aequilibrae_test_graph.aeg"))


def test_available_skims(setup_graph):
    setup_graph.prepare_graph(np.arange(5) + 1)
    avail = setup_graph.available_skims()
    data_fields = [
        "distance",
        "name",
        "lanes",
        "capacity",
        "speed",
        "b",
        "free_flow_time",
        "power",
        "colum",
        "volume",
        "modes",
    ]
    for i in data_fields:
        assert i in avail, "Skim availability with problems"


def test_compute_path(setup_graph):
    setup_graph.prepare_graph()
    setup_graph.set_graph("distance")
    setup_graph.set_blocked_centroid_flows(False)

    res = setup_graph.compute_path(1, 6)
    assert list(res.path) == [1, 4], "Number of path links is not correct"
    assert list(res.path_nodes) == [1, 2, 6], "Number of path nodes is not correct"


def test_compute_skims(setup_graph):
    setup_graph.prepare_graph()
    setup_graph.set_graph("distance")
    setup_graph.set_skimming(["distance", "free_flow_time"])
    setup_graph.set_blocked_centroid_flows(False)

    skm = setup_graph.compute_skims()
    skims = skm.results.skims
    assert skims.cores == 2, "Number of cores is not correct"
    assert skims.names == ["distance", "free_flow_time"], "Matrices names are not correct"


def test_exclude_links(setup_graph):
    setup_graph.set_blocked_centroid_flows(False)
    setup_graph.set_graph("distance")
    r1 = PathResults()
    r1.prepare(setup_graph)
    r1.compute_path(20, 21)
    assert list(r1.path) == [62]

    r1 = PathResults()
    setup_graph.exclude_links([62])
    r1.prepare(setup_graph)
    r1.compute_path(20, 21)
    assert list(r1.path) == [63, 69]