"""Centroid blocking with and without explicit turn penalties."""

import numpy as np
import pandas as pd

from aequilibrae import Graph
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.routing_context import make_routing_context


def shared_junction_graph():
    # Centroids 1 and 2 meet at ordinary junction 3. Centroid 5 connects
    # junctions 3 and 4, but must not serve as a shortcut between them.
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [13, 32, 14, 43, 35, 54, 36, 64],
            "a_node": [1, 3, 1, 4, 3, 5, 3, 6],
            "b_node": [3, 2, 4, 3, 5, 4, 6, 4],
            "direction": [1] * 8,
            "time": [1.0, 1.0, 2.0, 2.0, 0.1, 0.1, 2.0, 2.0],
        }
    )
    graph.prepare_graph(np.array([1, 2, 5]), remove_dead_ends=False)
    graph.set_graph("time")
    return graph


def test_centroid_blocking_without_turn_penalties_uses_node_search():
    graph = shared_junction_graph()
    graph.set_blocked_centroid_flows(True)

    assert isinstance(make_routing_context(graph), NodeBasedContext)
    assert list(graph.compute_path(1, 2).path_nodes) == [1, 3, 2]
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 6, 4]


def test_centroid_blocking_with_turn_penalty_only_bans_centroid_traversal():
    graph = shared_junction_graph()
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [1], "via_node": [4], "to_node": [3], "penalty": [0.5]})
    )
    graph.set_blocked_centroid_flows(False)
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 5, 4]

    graph.set_blocked_centroid_flows(True)
    assert isinstance(make_routing_context(graph), TurnBasedContext)
    assert list(graph.compute_path(1, 2).path_nodes) == [1, 3, 2]
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 6, 4]
