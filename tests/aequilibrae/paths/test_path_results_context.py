"""Path reporting from graph-independent search results."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Graph
from aequilibrae.paths.results import PathResults

from .test_assignment_integration import diamond


@pytest.mark.parametrize("early_exit", [False, True])
@pytest.mark.parametrize("penalty", [0.5, 10.0, np.inf])
def test_trace_uses_arrival_states_for_turn_paths_and_mileposts(early_exit, penalty):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [71, 55, 24, 12],
            "a_node": [10, 10, 30, 20],
            "b_node": [20, 30, 20, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0] * 4,
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    result = PathResults(graph, 10, 40, early_exit=early_exit)
    search = result.search_results
    destination = int(np.flatnonzero(result.node_ids == 40)[0])
    states = search.path_states_to(destination)
    np.testing.assert_array_equal(result.milepost, search.distances[states])
    if penalty == 0.5:
        np.testing.assert_array_equal(result.path, [71, 12])
        np.testing.assert_array_equal(result.path_nodes, [10, 20, 40])
        np.testing.assert_array_equal(result.milepost, [0.0, 1.0, 2.5])
    else:
        np.testing.assert_array_equal(result.path, [55, 24, 12])
        np.testing.assert_array_equal(result.path_nodes, [10, 30, 20, 40])
        np.testing.assert_array_equal(result.milepost, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.path_link_directions, np.ones(len(result.path)))

    result.update_trace(20)
    np.testing.assert_array_equal(result.path, [71])
    np.testing.assert_array_equal(result.milepost, [0.0, 1.0])
    result.update_trace(40)
    np.testing.assert_array_equal(result.milepost, search.distances[states])


@pytest.mark.parametrize("turn", [False, True])
def test_trace_distinguishes_intrazonal_and_missing_paths(turn):
    result = PathResults(diamond(turn), 10, 40)
    result.update_trace(10)
    assert result.path.size == result.path_link_directions.size == 0
    np.testing.assert_array_equal(result.path_nodes, [10])
    np.testing.assert_array_equal(result.milepost, [0.0])

    result.compute_path(40, 10)
    assert result.path is result.path_nodes is result.path_link_directions is result.milepost is None
    result.update_trace(40)
    assert result.path.size == 0
    np.testing.assert_array_equal(result.path_nodes, [40])
    np.testing.assert_array_equal(result.milepost, [0.0])
