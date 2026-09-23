import numpy as np
import pytest

from aequilibrae.paths import PathResults, available_heaps


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


@pytest.mark.parametrize("heuristic", ["haversine", "equirectangular"])
def test_path_results_support_astar(sioux_falls_example, heuristic):
    graph = build_graph(sioux_falls_example)
    result = PathResults(graph, 1, 20, a_star=True, heuristic=heuristic)

    assert heuristic in result.get_heuristics()
    assert result.path_nodes[0] == 1 and result.path_nodes[-1] == 20
    assert result.path is not None
