import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import PathResults, available_heaps, estimate_heuristic_scale
from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.a_star import EuclideanContext, a_star
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.results import path_results as path_results_module
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.workspaces import SearchWorkspace, AStarWorkspace

from .routing_helpers import make_context


def build_graph(project):
    project.network.build_graphs()
    graph = project.network.graphs["c"]
    graph.set_blocked_centroid_flows(False)
    graph.set_graph(cost_field="distance")
    graph.set_skimming("distance")
    return graph


@pytest.mark.parametrize("heap", available_heaps())
def test_path_results_are_identical_across_heaps(sioux_falls_example, heap):
    graph = build_graph(sioux_falls_example)
    reference = PathResults(graph, 1, 20)
    result = PathResults(graph, 1, 20, heap=heap)

    assert result.get_heaps() == available_heaps()
    result.set_heap(heap)
    result.compute_path(1, 20, heap=heap)
    assert result.path_nodes[0] == 1 and result.path_nodes[-1] == 20
    assert result.milepost[-1] == reference.milepost[-1]
    np.testing.assert_allclose(result.path_nodes, reference.path_nodes)
    np.testing.assert_allclose(result.skims.skims, reference.skims.skims)


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("turn", [False, True])
def test_new_dijkstra_heap_dispatch(heap, turn):
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 5, 2], turn=turn)
    query = SearchQuery(context.node_count, 0, np.array([False, False, True]))
    results = SearchResults(context.node_count, context.state_count, context.link_count)
    workspace = SearchWorkspace(context.node_count, context.state_count, heap=heap)
    dijkstra(context, query, results, workspace)
    np.testing.assert_array_equal(results.path_links_to(2), [0, 2])
    assert results.path_cost_to(2) == 3
    with pytest.raises(ValueError, match="heap must be one of"):
        SearchWorkspace(context.node_count, context.state_count, heap="invalid")


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("turn", [False, True])
def test_separate_workspaces_reuse_their_heaps(heap, turn):
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 5, 2], turn=turn)
    results = SearchResults(context.node_count, context.state_count, context.link_count)
    search = SearchWorkspace(context.node_count, context.state_count, heap=heap)
    astar = AStarWorkspace(context.node_count, context.state_count, heap=heap)
    heuristic = EuclideanContext([0, 1, 2], [0, 0, 0], 0)
    assert search.heap == astar.heap == heap
    assert not isinstance(search, AStarWorkspace)

    for destination in (2, 1, 2):
        mask = np.zeros(context.node_count, dtype=bool)
        mask[destination] = True
        query = SearchQuery(context.node_count, 0, mask)
        assert dijkstra(context, query, results, search).path_cost_to(destination) == (3 if destination == 2 else 1)
        assert a_star(context, query, destination, heuristic, results, astar).path_cost_to(destination) == (
            3 if destination == 2 else 1
        )

    with pytest.raises(ValueError, match="workspace dimensions"):
        dijkstra(context, query, results, SearchWorkspace(context.node_count, context.state_count + 1))
    with pytest.raises(ValueError, match="workspace dimensions"):
        a_star(
            context, query, destination, heuristic, results, AStarWorkspace(context.node_count + 1, context.state_count)
        )


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("turn", [False, True])
def test_prepared_aon_heap_dispatch(heap, turn):
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 5, 2], turn=turn)
    demand = np.zeros((3, 3, 1))
    demand[0, 2, 0] = 7
    prepared = PreparedAoN(context, demand, heap=heap, cores=2)
    output = prepared.make_outputs()
    prepared.run(output)
    np.testing.assert_array_equal(output.loading.link_loads[:, 0], [7, 0, 7])
    prepared.run(output)
    np.testing.assert_array_equal(output.loading.link_loads[:, 0], [7, 0, 7])
    with pytest.raises(ValueError, match="heap must be one of"):
        PreparedAoN(context, demand, heap="invalid")


@pytest.mark.parametrize("heap", available_heaps())
def test_path_results_remembers_selected_heap(sioux_falls_example, heap, monkeypatch):
    graph = build_graph(sioux_falls_example)
    result = PathResults(graph, 1, 20, heap=heap)
    selected = []

    def recording_dijkstra(context, query, results, workspace):
        selected.append(workspace.heap)
        return dijkstra(context, query, results, workspace)

    monkeypatch.setattr(path_results_module, "dijkstra", recording_dijkstra)
    assert result._heap == heap
    result.compute_path(1, 20, heap="4ary")
    assert result._heap == heap  # Per-call overrides do not change the selection.
    result.compute_path(1, 20)
    assert selected == ["4ary", heap]
    result.set_heap("std")
    result.reset()
    assert result._heap == "std"
    result.compute_path(1, 20)
    assert selected[-1] == "std"
    with pytest.raises(ValueError, match="heap must be one of"):
        result.set_heap("unknown")
    with pytest.raises(ValueError, match="heap must be one of"):
        result.compute_path(1, 20, heap="unknown")


@pytest.mark.parametrize("heuristic", ["haversine", "euclidean"])
def test_path_results_support_astar(sioux_falls_example, heuristic):
    graph = build_graph(sioux_falls_example)
    lonlat = graph.lonlat_index
    points = gpd.GeoSeries(gpd.points_from_xy(lonlat.lon, lonlat.lat), index=lonlat.index, crs=4326)
    points = points.to_crs(points.estimate_utm_crs())
    coordinates = pd.DataFrame({"x": points.x, "y": points.y})
    scale = estimate_heuristic_scale(graph, coordinates, heuristic=heuristic)
    result = PathResults(graph, 1, 20, a_star=True, heuristic=heuristic, coordinates=coordinates, heuristic_scale=scale)
    reference = PathResults(graph, 1, 20)
    assert result.milepost[-1] == pytest.approx(reference.milepost[-1])

    assert heuristic in result.get_heuristics()
    assert result.path_nodes[0] == 1 and result.path_nodes[-1] == 20
    assert result.path is not None
