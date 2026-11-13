"""Centroid blocking with and without explicit turn penalties."""

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext

from aequilibrae.paths import Network

from .routing_helpers import search


def shared_junction_network():
    # Centroids 1 and 2 meet at ordinary junction 3. Centroid 5 connects
    # junctions 3 and 4, but must not serve as a shortcut between them.
    links = pd.DataFrame(
        {
            "link_id": [13, 32, 14, 43, 35, 54, 36, 64],
            "a_node": [1, 3, 1, 4, 3, 5, 3, 6],
            "b_node": [3, 2, 4, 3, 5, 4, 6, 4],
            "direction": [1] * 8,
            "time": [1.0, 1.0, 2.0, 2.0, 0.1, 0.1, 2.0, 2.0],
        }
    )
    return Network(links, centroids=[1, 2, 5])


@pytest.mark.parametrize("compact", [False, True])
def test_centroid_blocking_without_turn_penalties_uses_node_search(compact):
    graph = shared_junction_network().build_graph("c", "time", compact=compact)
    assert isinstance(graph.routing, NodeBasedContext)
    assert list(graph.compute_path(1, 2).path_nodes) == [1, 3, 2]
    path = graph.compute_path(3, 4)
    np.testing.assert_array_equal(path.get_network_links()[0], [36, 64])
    assert list(path.path_nodes) == ([3, 4] if compact else [3, 6, 4])


@pytest.mark.parametrize("compact", [False, True])
def test_centroid_blocking_with_turn_penalty_only_bans_centroid_traversal(compact):
    network = shared_junction_network()
    turns = pd.DataFrame({"from_node": [1], "via_node": [4], "to_node": [3], "penalty": [0.5]})
    graph = network.build_graph("c", "time", compact=compact, turn_restrictions=turns, block_centroid_flows=False)
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 5, 4]

    graph = network.build_graph("c", "time", compact=compact, turn_restrictions=turns)
    assert isinstance(graph.routing, TurnBasedContext)
    assert list(graph.compute_path(1, 2).path_nodes) == [1, 3, 2]
    path = graph.compute_path(3, 4)
    np.testing.assert_array_equal(path.get_network_links()[0], [36, 64])
    assert list(path.path_nodes) == ([3, 4] if compact else [3, 6, 4])

    from_1, from_3, to_2, to_4 = graph.mapping.graph_node([1, 3, 2, 4])
    assert search(graph.routing, from_1).path_cost_to(to_2) == 2.0
    assert search(graph.routing, from_3).path_cost_to(to_4) == 4.0


@pytest.mark.parametrize("allow_uturns", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_single_bidirectional_connector_bans_centroid_traversal(allow_uturns, compact):
    links = pd.DataFrame(
        {
            "link_id": [13, 34, 36],
            "a_node": [1, 3, 3],
            "b_node": [3, 4, 6],
            "direction": [0, 1, 1],
            "time": [1.0, 1.0, 1.0],
        }
    )
    turns = pd.DataFrame({"from_node": [1], "via_node": [3], "to_node": [4], "penalty": [0.5]})
    graph = Network(links, centroids=[1], turn_restrictions=turns, allow_uturns=allow_uturns).build_graph(
        "c", "time", compact=compact
    )
    context = graph.routing
    into_centroid = graph.mapping.graph_link(13, -1)
    out_of_centroid = graph.mapping.graph_link(13, 1)
    first, last = context.turn_fs[into_centroid : into_centroid + 2]
    bans = dict(zip(context.turn_to_links[first:last], context.turn_penalties[first:last], strict=True))

    # Explicit finite turn penalties can permit U-turns even when other U-turns
    # are disabled, so centroid traversal needs an explicit ban in both cases.
    assert np.isinf(bans[out_of_centroid])
    assert list(graph.compute_path(1, 4).path_nodes) == [1, 3, 4]
