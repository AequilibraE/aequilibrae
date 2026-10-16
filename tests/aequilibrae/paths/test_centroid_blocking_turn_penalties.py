"""Centroid blocking with and without explicit turn penalties."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Graph
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.routing_context import make_routing_context

from .routing_helpers import search


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
    graph.set_turn_restrictions(pd.DataFrame({"from_node": [1], "via_node": [4], "to_node": [3], "penalty": [0.5]}))
    graph.set_blocked_centroid_flows(False)
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 5, 4]

    graph.set_blocked_centroid_flows(True)
    assert isinstance(make_routing_context(graph), TurnBasedContext)
    assert list(graph.compute_path(1, 2).path_nodes) == [1, 3, 2]
    assert list(graph.compute_path(3, 4).path_nodes) == [3, 6, 4]

    compact = make_routing_context(graph, compact=True)
    assert isinstance(compact, TurnBasedContext)
    from_1 = graph.compact_nodes_to_indices[1]
    from_3 = graph.compact_nodes_to_indices[3]
    to_2 = graph.compact_nodes_to_indices[2]
    to_4 = graph.compact_nodes_to_indices[4]
    assert search(compact, from_1).path_cost_to(to_2) == 2.0
    assert search(compact, from_3).path_cost_to(to_4) == 4.0


@pytest.mark.parametrize("allow_path_uturns", [False, True])
def test_single_bidirectional_connector_only_needs_ban_when_uturns_allowed(allow_path_uturns):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [13, 34, 36],
            "a_node": [1, 3, 3],
            "b_node": [3, 4, 6],
            "direction": [0, 1, 1],
            "time": [1.0, 1.0, 1.0],
        }
    )
    graph.prepare_graph(np.array([1]), remove_dead_ends=False)
    graph.set_graph("time")
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [1], "via_node": [3], "to_node": [4], "penalty": [0.5]}),
        allow_path_uturns=allow_path_uturns,
    )
    graph.set_blocked_centroid_flows(True)

    for compact in (False, True):
        links = graph.compact_graph if compact else graph.graph
        node_ids = graph.compact_nodes_to_indices if compact else graph.nodes_to_indices
        offsets = graph.compact_turn_fs if compact else graph.turn_fs
        to_arcs = graph.compact_turn_to_arcs if compact else graph.turn_to_arcs
        penalties = graph.compact_turn_penalties if compact else graph.turn_penalties
        arcs = links.set_index(["a_node", "b_node"])["id"]
        into_centroid = int(arcs.loc[(node_ids[3], node_ids[1])])
        out_of_centroid = int(arcs.loc[(node_ids[1], node_ids[3])])
        first, last = offsets[into_centroid : into_centroid + 2]
        bans = {int(arc): penalty for arc, penalty in zip(to_arcs[first:last], penalties[first:last], strict=True)}
        if allow_path_uturns:
            assert np.isinf(bans[out_of_centroid])
        else:
            assert out_of_centroid not in bans

    assert list(graph.compute_path(1, 4).path_nodes) == [1, 3, 4]
