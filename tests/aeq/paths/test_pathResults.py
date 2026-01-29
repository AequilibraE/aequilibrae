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
    assert r.origin is None
    assert r.destination is None
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


def test_compute_paths(p_results):
    r = p_results["r"]

    for early_exit, a_star in product([True, False], repeat=2):
        r.early_exit = early_exit
        r.a_star = a_star
        path_computation(5, 2, p_results["g"], r)
        assert list(r.path) == [12, 14]
        assert list(r.path_link_directions) == [1, 1]
        assert list(r.path_nodes) == [5, 6, 2]
        assert list(r.milepost) == [0, 4, 9]


def test_compute_with_skimming(p_results):
    g = p_results["g"]
    for early_exit in [True, False]:
        r = PathResults()
        g.set_skimming("free_flow_time")
        r.prepare(g)
        r.compute_path(origin, dest, early_exit=early_exit)
        assert r.milepost[-1] == r.skims[dest]


def test_update_trace(p_results):
    r = p_results["r"]

    for early_exit, a_star in product([True, False], repeat=2):
        r.compute_path(origin, 2, early_exit=early_exit, a_star=a_star)
        r.update_trace(10)
        assert list(r.path) == [13, 25]
        assert list(r.path_link_directions) == [1, 1]
        assert list(r.path_nodes) == [5, 9, 10]
        assert list(r.milepost) == [0, 5, 8]


def test_compute_path_optimization(p_results):
    """Test that compute_path optimization logic correctly reuses computations when appropriate"""
    r = p_results["r"]
    g = p_results["g"]
    
    # Test 1: Same origin, same settings, different destinations should reuse computation
    r.compute_path(origin, 2, early_exit=False, a_star=False)
    path_to_2 = list(r.path)
    nodes_to_2 = list(r.path_nodes)
    predecessors_after_first = r.predecessors.copy()
    
    # Compute path to different destination with same origin and settings
    # This should trigger update_trace, not full recomputation
    r.compute_path(origin, 10, early_exit=False, a_star=False)
    path_to_10 = list(r.path)
    nodes_to_10 = list(r.path_nodes)
    predecessors_after_second = r.predecessors.copy()
    
    # Predecessors should be identical (no recomputation occurred)
    assert np.array_equal(predecessors_after_first, predecessors_after_second)
    # But paths should be different
    assert path_to_2 != path_to_10
    assert nodes_to_2 != nodes_to_10
    
    # Test 2: Changing early_exit should trigger recomputation
    r.reset()
    r.compute_path(origin, 2, early_exit=False, a_star=False)
    predecessors_full = r.predecessors.copy()
    
    # Use a more distant destination to ensure different exploration patterns
    r.compute_path(origin, dest, early_exit=True, a_star=False)
    predecessors_early_exit = r.predecessors.copy()
    
    # With early_exit to a distant destination, fewer nodes should be explored
    # Count of non-(-1) predecessors should differ
    assert np.count_nonzero(predecessors_full != -1) > np.count_nonzero(predecessors_early_exit != -1)
    
    # Test 3: Changing a_star should trigger recomputation
    r.reset()
    r.compute_path(origin, 2, early_exit=False, a_star=False)
    predecessors_no_astar = r.predecessors.copy()
    
    r.compute_path(origin, 2, early_exit=False, a_star=True, heuristic="haversine")
    predecessors_with_astar = r.predecessors.copy()
    
    # With A* enabled, the exploration pattern may differ (A* uses heuristic)
    # Verify both computations completed successfully
    assert r.path is not None
    # A* may explore fewer nodes due to heuristic guidance
    assert np.count_nonzero(predecessors_no_astar != -1) >= np.count_nonzero(predecessors_with_astar != -1)
    
    # Test 4: Changing heuristic should trigger recomputation
    r.reset()
    r.compute_path(origin, dest, early_exit=False, a_star=True, heuristic="haversine")
    predecessors_haversine = r.predecessors.copy()
    path_haversine = list(r.path)
    
    r.compute_path(origin, dest, early_exit=False, a_star=True, heuristic="equirectangular")
    predecessors_equirectangular = r.predecessors.copy()
    path_equirectangular = list(r.path)
    
    # Different heuristics may result in different exploration patterns
    # Both should produce valid paths (though they may be the same shortest path)
    assert r.path is not None
    assert path_haversine == path_equirectangular  # Shortest path should be the same
    
    # Test 5: Same heuristic (or None) should not trigger recomputation of algorithm
    # Note: With early_exit, if destination was not found in previous tree, recomputation
    # is necessary but uses update_trace path which may call compute_path internally
    r.reset()
    r.compute_path(origin, 2, early_exit=False, a_star=True, heuristic="haversine")
    
    # Call again with same heuristic (None should use current)
    # This should use update_trace optimization path
    r.compute_path(origin, 2, early_exit=False, a_star=True, heuristic=None)
    
    # Since we're asking for the same destination and settings, no full recomputation needed
    # The path should be valid
    assert r.path is not None


def test_compute_path_with_skimming_optimization(p_results):
    """Test that skims are correctly updated through the update_trace path"""
    g = p_results["g"]
    g.set_skimming("free_flow_time")
    
    r = PathResults()
    r.prepare(g)
    
    # Compute path to first destination
    r.compute_path(origin, 2, early_exit=False)
    skim_value_2 = r.skims[2]
    assert r.milepost[-1] == skim_value_2
    
    # Compute path to second destination with same origin
    # This should use update_trace optimization
    r.compute_path(origin, 10, early_exit=False)
    skim_value_10 = r.skims[10]
    assert r.milepost[-1] == skim_value_10
    
    # Verify skim for first destination is still correct
    assert r.skims[2] == skim_value_2
    
    # Compute to a third destination
    r.compute_path(origin, dest, early_exit=False)
    skim_value_dest = r.skims[dest]
    assert r.milepost[-1] == skim_value_dest
    
    # All previous skims should still be correct
    assert r.skims[2] == skim_value_2
    assert r.skims[10] == skim_value_10


# --- Blocking triangle network tests ---


@pytest.fixture(scope="function")
def triangle_blocking_setup(triangle_graph_blocking):
    triangle_graph_blocking.network.build_graphs(modes=["c"])
    g = triangle_graph_blocking.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(True)
    r = PathResults()
    r.prepare(g)
    return {"project": triangle_graph_blocking, "g": g, "r": r}


def test_triangle_compute_paths(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    for early_exit, a_star in product([True, False], repeat=2):
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


def test_triangle_compute_blocking_paths(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    for early_exit, a_star in product([True, False], repeat=2):
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


def test_triangle_update_trace(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    for early_exit, a_star in product([True, False], repeat=2):
        r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
        assert list(r.path_nodes) == [1, 3, 2]
        assert list(r.path) == [1, 2]
        r.update_trace(3)
        assert list(r.path_nodes) == [1, 3]
        assert list(r.path) == [1]


def test_triangle_update_blocking_trace(triangle_blocking_setup):
    r = triangle_blocking_setup["r"]
    for early_exit, a_star in product([True, False], repeat=2):
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
