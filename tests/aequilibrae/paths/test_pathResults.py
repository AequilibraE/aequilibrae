import zipfile
from itertools import product
from os.path import join
from tempfile import gettempdir

import numpy as np
import pytest

from aequilibrae import Project
from aequilibrae.paths import path_computation
from aequilibrae.paths.results import PathResults

origin = 5
dest = 13


@pytest.fixture(scope="function")
def p_results(sioux_falls_example):
    project = sioux_falls_example
    project.network.build_graphs()
    g = project.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    r = PathResults()
    r.prepare(g)
    yield {"project": project, "g": g, "matrix": matrix, "r": r}
    project.close()
    matrix.close()
    del r


def test_reset(p_results):
    r = p_results["r"]
    r.compute_path(dest, origin, early_exit=True, a_star=True, heuristic="haversine")
    r.reset()
    assert r.path is None
    assert r.path_nodes is None
    assert r.path_link_directions is None
    assert r.milepost is None
    assert r.predecessors.max() == -1
    assert r.predecessors.min() == -1
    assert r.connectors.max() == -1
    assert r.connectors.min() == -1
    assert r.early_exit is False
    assert r._early_exit is False
    assert r.a_star is False
    assert r._a_star is False
    assert r._heuristic == "equirectangular"
    if r.skims is not None:
        assert r.skims.max() == np.inf
        assert r.skims.min() == np.inf
    new_r = PathResults()
    with pytest.raises(ValueError):
        new_r.reset()


def test_heuristics(p_results):
    r = p_results["r"]
    assert r.get_heuristics() == ["haversine", "equirectangular"]
    r.set_heuristic("haversine")
    assert r._heuristic == "haversine"
    r.set_heuristic("equirectangular")
    assert r._heuristic == "equirectangular"


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_compute_paths(p_results, early_exit, a_star):
    r = p_results["r"]
    r.early_exit = early_exit
    r.a_star = a_star
    path_computation(5, 2, p_results["g"], r)
    assert list(r.path) == [12, 14]
    assert list(r.path_link_directions) == [1, 1]
    assert list(r.path_nodes) == [5, 6, 2]
    assert list(r.milepost) == [0, 4, 9]


@pytest.mark.parametrize("early_exit", [True, False])
def test_compute_with_skimming(p_results, early_exit):
    g = p_results["g"]
    r = PathResults()
    g.set_skimming("free_flow_time")
    r.prepare(g)
    r.compute_path(origin, dest, early_exit=early_exit)
    assert r.milepost[-1] == r.skims[dest]


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_update_trace(p_results, early_exit, a_star):
    r = p_results["r"]
    r.compute_path(origin, 2, early_exit=early_exit, a_star=a_star)
    r.update_trace(10)
    assert list(r.path) == [13, 25]
    assert list(r.path_link_directions) == [1, 1]
    assert list(r.path_nodes) == [5, 9, 10]
    assert list(r.milepost) == [0, 5, 8]


# --- Blocking triangle network tests ---


@pytest.fixture(scope="function")
def triangle_blocking_setup(triangle_graph_blocking):
    triangle_graph_blocking.network.build_graphs(modes=["c"])
    g = triangle_graph_blocking.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(True)
    r = PathResults()
    r.prepare(g)
    yield {"project": triangle_graph_blocking, "g": g, "r": r}
    triangle_graph_blocking.close()
    del r


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_triangle_compute_paths(triangle_blocking_setup, early_exit, a_star):
    r = triangle_blocking_setup["r"]
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [1, 3, 2]
    assert list(r.path) == [1, 2]
    r.compute_path(2, 1, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [2, 1]
    assert list(r.path) == [3]
    r.compute_path(3, 1, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [3, 2, 1]
    assert list(r.path) == [2, 3]
    r.compute_path(3, 2, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [3, 2]
    assert list(r.path) == [2]
    r.compute_path(1, 3, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [1, 3]
    assert list(r.path) == [1]
    r.compute_path(2, 3, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [2, 1, 3]
    assert list(r.path) == [3, 1]


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_triangle_compute_blocking_paths(triangle_blocking_setup, early_exit, a_star):
    r = triangle_blocking_setup["r"]
    r.compute_path(4, 5, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [4, 1, 3, 2, 5]
    assert list(r.path) == [4, 1, 2, 5]
    r.compute_path(5, 4, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [5, 2, 1, 4]
    assert list(r.path) == [5, 3, 4]
    r.compute_path(6, 4, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [6, 3, 2, 1, 4]
    assert list(r.path) == [6, 2, 3, 4]
    r.compute_path(6, 5, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [6, 3, 2, 5]
    assert list(r.path) == [6, 2, 5]
    r.compute_path(4, 6, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [4, 1, 3, 6]
    assert list(r.path) == [4, 1, 6]
    r.compute_path(5, 6, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [5, 2, 1, 3, 6]
    assert list(r.path) == [5, 3, 1, 6]


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_triangle_update_trace(triangle_blocking_setup, early_exit, a_star):
    r = triangle_blocking_setup["r"]
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [1, 3, 2]
    assert list(r.path) == [1, 2]
    r.update_trace(3)
    assert list(r.path_nodes) == [1, 3]
    assert list(r.path) == [1]


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_triangle_update_blocking_trace(triangle_blocking_setup, early_exit, a_star):
    r = triangle_blocking_setup["r"]
    r.compute_path(4, 5, early_exit=early_exit, a_star=a_star)
    assert list(r.path_nodes) == [4, 1, 3, 2, 5]
    assert list(r.path) == [4, 1, 2, 5]
    r.update_trace(6)
    assert list(r.path_nodes) == [4, 1, 3, 6]
    assert list(r.path) == [4, 1, 6]


def test_triangle_update_trace_early_exit(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    r.compute_path(1, 6, early_exit=True)
    assert list(r.path_nodes) == [1, 3, 6]
    assert list(r.path) == [1, 6]
    assert [r.graph.all_nodes[x] if x != -1 else -1 for x in r.predecessors] == [1, -1, 3, -1, -1, 1, -1]
    r.early_exit = True
    r.update_trace(2)
    assert list(r.path_nodes) == [1, 3, 2]
    assert list(r.path) == [1, 2]
    assert [r.graph.all_nodes[x] if x != -1 else -1 for x in r.predecessors] == [1, -1, 3, -1, 3, 1, -1]


def test_triangle_update_trace_full(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    r.compute_path(1, 6, early_exit=True)
    r.early_exit = False
    r.update_trace(2)
    assert list(r.path_nodes) == [1, 3, 2]
    assert list(r.path) == [1, 2]
    assert [r.graph.all_nodes[x] if x != -1 else -1 for x in r.predecessors] == [1, 2, 3, -1, 3, 1, -1]


def test_compute_paths_centroid_last_node_id(test_data_path):
    zipfile.ZipFile(test_data_path / "St_Varent_issue307.zip").extractall(gettempdir())
    st_varent = join(gettempdir(), "St_Varent")
    project = Project()
    project.open(st_varent)
    project.network.build_graphs()
    g = project.network.graphs["c"]
    g.set_graph("distance")
    g.set_skimming("distance")
    r = PathResults()
    r.prepare(g)
    r.compute_path(387, 1067)
    project.close()
