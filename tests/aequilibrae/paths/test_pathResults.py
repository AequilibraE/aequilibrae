import numpy as np
import pytest

from aequilibrae.paths import available_heaps
from aequilibrae.paths.results import PathResults

origin = 5
dest = 13


@pytest.fixture(scope="function")
def p_results(sioux_falls_example):
    project = sioux_falls_example
    project.network.build_graphs()

    graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)

    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()

    yield {"project": project, "g": graph, "matrix": matrix}

    matrix.close()


def test_reset(p_results):
    result = PathResults(p_results["g"], origin, dest, early_exit=True)
    result.reset()

    assert result.path is None
    assert result.path_nodes is None
    assert result.path_link_directions is None
    assert result.milepost is None
    assert np.all(result.predecessors == result.search_results.sentinel)
    assert np.all(result.connectors == result.search_results.sentinel)
    assert result.origin is None
    assert result.destination is None
    assert result.early_exit is False
    if result.skims is not None:
        assert np.all(np.isinf(result.skims.skims))


def test_compute_paths(p_results):
    graph = p_results["g"]
    result = PathResults(graph, origin, dest, early_exit=True)

    for early_exit in [True, False]:
        for a_star, heuristic in [(False, None), (True, "haversine")]:
            result.compute_path(5, 2, early_exit=early_exit, a_star=a_star, heuristic=heuristic)
            assert list(result.path) == [12, 14]
            assert list(result.path_link_directions) == [1, 1]
            assert list(result.path_nodes) == [5, 6, 2]
            assert list(result.milepost) == [0, 4, 9]


@pytest.mark.parametrize("heap", available_heaps())
def test_compute_with_skimming(p_results, heap):
    graph = p_results["g"]
    graph.set_skimming("free_flow_time")
    for early_exit in [True, False]:
        result = PathResults(graph, origin, dest, early_exit=early_exit, heap=heap)
        destination = int(np.flatnonzero(result.node_ids == 13)[0])
        assert result.milepost[-1] == result.skims.matrices["free_flow_time"][0, destination]


@pytest.mark.parametrize("heap", available_heaps())
def test_update_trace(p_results, heap):
    graph = p_results["g"]
    for early_exit in [True, False]:
        result = PathResults(graph, origin, 2, early_exit=early_exit, heap=heap)
        result.update_trace(10)
        assert list(result.path) == [13, 25]
        assert list(result.path_link_directions) == [1, 1]
        assert list(result.path_nodes) == [5, 9, 10]
        assert list(result.milepost) == [0, 5, 8]


@pytest.fixture(scope="function")
def triangle_blocking_setup(triangle_graph_blocking):
    triangle_graph_blocking.network.build_graphs(modes=["c"])
    graph = triangle_graph_blocking.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(True)
    return {"project": triangle_graph_blocking, "g": graph}


def test_triangle_compute_paths(triangle_blocking_setup):
    graph = triangle_blocking_setup["g"]
    for early_exit in [True, False]:
        expected = [
            (1, 2, [1, 3, 2], [1, 2]),
            (2, 1, [2, 1], [3]),
            (3, 1, [3, 2, 1], [2, 3]),
            (3, 2, [3, 2], [2]),
            (1, 3, [1, 3], [1]),
            (2, 3, [2, 1, 3], [3, 1]),
        ]
        for origin_id, destination, nodes, links in expected:
            result = PathResults(graph, origin_id, destination, early_exit=early_exit)
            assert list(result.path_nodes) == nodes
            assert list(result.path) == links


def test_triangle_compute_blocking_paths(triangle_blocking_setup):
    graph = triangle_blocking_setup["g"]
    expected = [
        (4, 5, [4, 1, 3, 2, 5], [4, 1, 2, 5]),
        (5, 4, [5, 2, 1, 4], [5, 3, 4]),
        (6, 4, [6, 3, 2, 1, 4], [6, 2, 3, 4]),
        (6, 5, [6, 3, 2, 5], [6, 2, 5]),
        (4, 6, [4, 1, 3, 6], [4, 1, 6]),
        (5, 6, [5, 2, 1, 3, 6], [5, 3, 1, 6]),
    ]
    for early_exit in [True, False]:
        for origin_id, destination, nodes, links in expected:
            result = PathResults(graph, origin_id, destination, early_exit=early_exit)
            assert list(result.path_nodes) == nodes
            assert list(result.path) == links


def test_triangle_update_trace(triangle_blocking_setup):
    graph = triangle_blocking_setup["g"]
    for early_exit in [True, False]:
        result = PathResults(graph, 1, 2, early_exit=early_exit)
        assert list(result.path_nodes) == [1, 3, 2]
        assert list(result.path) == [1, 2]
        result.update_trace(3)
        assert list(result.path_nodes) == [1, 3]
        assert list(result.path) == [1]


def test_triangle_update_blocking_trace(triangle_blocking_setup):
    graph = triangle_blocking_setup["g"]
    for early_exit in [True, False]:
        result = PathResults(graph, 4, 5, early_exit=early_exit)
        assert list(result.path_nodes) == [4, 1, 3, 2, 5]
        assert list(result.path) == [4, 1, 2, 5]
        result.update_trace(6)
        assert list(result.path_nodes) == [4, 1, 3, 6]
        assert list(result.path) == [4, 1, 6]


def test_triangle_update_trace_early_exit(triangle_blocking_setup):
    result = PathResults(triangle_blocking_setup["g"], 1, 6, early_exit=True)
    assert list(result.path_nodes) == [1, 3, 6]
    assert list(result.path) == [1, 6]
    result.update_trace(2)
    assert list(result.path_nodes) == [1, 3, 2]
    assert list(result.path) == [1, 2]


def test_triangle_update_trace_full(triangle_blocking_setup):
    result = PathResults(triangle_blocking_setup["g"], 1, 6, early_exit=False)
    assert list(result.path_nodes) == [1, 3, 6]
    assert list(result.path) == [1, 6]
    result.update_trace(2)
    assert list(result.path_nodes) == [1, 3, 2]
    assert list(result.path) == [1, 2]


def test_compute_paths_centroid_last_node_id(st_varent):
    st_varent.network.build_graphs()
    graph = st_varent.network.graphs["c"]
    graph.set_graph("distance")
    graph.set_skimming("distance")
    PathResults(graph, 387, 1067)
