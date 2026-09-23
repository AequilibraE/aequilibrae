import numpy as np
import pytest

from aequilibrae.paths import PathResults, available_heaps
from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.results import path_results as path_results_module
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults

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
    dijkstra(context, query, results, heap=heap)
    np.testing.assert_array_equal(results.path_links_to(2), [0, 2])
    assert results.path_cost_to(2) == 3
    with pytest.raises(ValueError, match="heap must be one of"):
        dijkstra(context, query, results, heap="invalid")


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

    def recording_dijkstra(context, query, results, heap):
        selected.append(heap)
        return dijkstra(context, query, results, heap=heap)

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


@pytest.mark.parametrize("heuristic", ["haversine", "equirectangular"])
def test_path_results_support_astar(sioux_falls_example, heuristic):
    graph = build_graph(sioux_falls_example)
    result = PathResults(graph, 1, 20, a_star=True, heuristic=heuristic)

    assert heuristic in result.get_heuristics()
    assert result.path_nodes[0] == 1 and result.path_nodes[-1] == 20
    assert result.path is not None
