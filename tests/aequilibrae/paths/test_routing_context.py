"""Shared routing setup, snapshots and external IDs."""

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext

from aequilibrae.paths.graph import GraphBase, freeze_graph
from aequilibrae.paths.routing_context import GraphMapping, make_routing_context

from .routing_helpers import make_context, search
from .test_assignment_integration import diamond


def chain_graph(centroids=True):
    graph = GraphBase()
    graph.network = pd.DataFrame(
        {
            "link_id": [71, 12, 55],
            "a_node": [10, 20, 30],
            "b_node": [20, 30, 40],
            "direction": [0, 0, 0],
            "time": [1.0, 2.0, 3.0],
        }
    )
    graph.prepare_graph(np.array([10, 40]) if centroids else None, remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")

    return graph


def compressed_uturn_graph(with_return_chain=True):
    r"""Ban 1 -> 2 -> 6 so reaching 6 requires returning to 2 with another arrival.

    Physical graph:

        1 -> 2 -> 6
             | \
             4  5
             | /
             3

    The chain 2 <-> 4 <-> 3 is bidirectional. The optional chain 3 -> 5 -> 2
    provides a legal return; reversing 3 -> 4 is a banned U-turn.
    All link costs are 1. Chain compression removes the internal nodes.
    """
    graph = GraphBase()
    graph.network = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4, 5, 6],
            "a_node": [1, 2, 4, 3, 5, 2],
            "b_node": [2, 4, 3, 5, 2, 6],
            "direction": [1, 0, 0, 1, 1, 1],
            "time": [1.0] * 6,
        }
    )

    # Drops links 4 & 5
    if not with_return_chain:
        graph.network = graph.network.loc[~graph.network.link_id.isin([4, 5])].copy()

    graph.prepare_graph(np.array([1, 3, 6]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_turn_restrictions(pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [6], "penalty": [np.inf]}))
    return graph


def test_compact_dijkstra_allows_return_through_another_chain():
    graph_base = compressed_uturn_graph()
    assert 4 not in graph_base.compact_all_nodes
    assert 5 not in graph_base.compact_all_nodes
    
    # Build a Graph for path computation
    graph = freeze_graph(graph_base, compact=False)
    np.testing.assert_array_equal(graph.compute_path(1, 6).path_nodes, [1, 2, 4, 3, 5, 2, 6])

    # Test compact routing context (internal infrastructure test)
    context = make_routing_context(graph_base, compact=True)
    origin = graph_base.compact_nodes_to_indices[1]
    destination = graph_base.compact_nodes_to_indices[6]
    results = search(context, origin, destination)

    # Links 2/3 form the outward chain and links 4/5 are the return via 5.
    forward = graph_base.graph.loc[graph_base.graph.direction == 1].set_index("link_id")
    expected_links = forward.loc[[1, 2, 4, 6], "__compressed_id__"].to_numpy()
    np.testing.assert_array_equal(results.path_links_to(destination), expected_links)
    assert results.path_cost_to(destination) == 6.0


def test_compact_dijkstra_bans_reversal_through_same_chain():
    graph_base = compressed_uturn_graph(with_return_chain=False)

    # Without 3 -> 5 -> 2, the only return is 3 -> 4 -> 3, so 6 is unreachable.
    graph = freeze_graph(graph_base, compact=False)
    assert graph.compute_path(1, 6).path is None

    # Check compact as well
    context = make_routing_context(graph_base, compact=True)
    origin = graph_base.compact_nodes_to_indices[1]
    destination = graph_base.compact_nodes_to_indices[6]
    results = search(context, origin, destination)
    assert not results.reachable_to(destination)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_routing_context_copies_graph_inputs(turn, compact):
    graph = diamond(turn)
    context = make_routing_context(graph, compact=compact)
    expected = search(context, 0).path_cost_to(3)

    assert isinstance(context, TurnBasedContext if turn else NodeBasedContext)
    assert expected == (2.5 if turn else 2.0)

    offsets = graph.compact_fs if compact else graph.fs
    links = graph.compact_graph if compact else graph.graph
    costs = graph.compact_cost if compact else graph.cost

    assert not np.shares_memory(context.fs, offsets)
    assert not np.shares_memory(context.heads, links.b_node.to_numpy())
    assert not np.shares_memory(context.costs, costs)

    offsets[:] = 0
    links.loc[:, "b_node"] = 0
    costs[:] = 100

    if turn:
        penalties = graph.compact_turn_penalties if compact else graph.turn_penalties
        assert not np.shares_memory(context.turn_penalties, penalties)
        penalties[:] = 100

    assert search(context, 0).path_cost_to(3) == expected


@pytest.mark.parametrize("compact", [False, True])
def test_supplied_cost_buffer_is_borrowed(compact):
    graph = diamond()
    costs = np.ones(graph.compact_num_links if compact else graph.num_links)
    original = graph.cost.copy()
    context = make_routing_context(graph, costs, compact=compact)

    assert np.shares_memory(context.costs, costs)
    assert costs.flags.writeable
    assert search(context, 0).path_cost_to(3) == 2.0

    costs[:] = 3

    assert search(context, 0).path_cost_to(3) == 6.0
    np.testing.assert_array_equal(graph.cost, original)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_centroid_blocking_keeps_assignment_rules(turn, compact):
    graph = diamond(turn)
    graph.set_blocked_centroid_flows(True)
    context = make_routing_context(graph, compact=compact)

    assert context.blocked_centroid_count == (0 if graph.has_turn_restrictions else graph.num_zones)


@pytest.mark.parametrize("centroids", [False, True])
def test_full_graph_keeps_intermediate_nodes(centroids):
    graph = chain_graph(centroids)
    context = make_routing_context(graph)
    mapping = GraphMapping(context, graph.all_nodes, graph.graph.link_id, graph.graph.direction)
    origin = mapping.node_index(20)
    destination = mapping.node_index(30)
    results = search(context, origin)
    links = results.path_links_to(destination)

    assert context.node_count == 4
    assert results.path_cost_to(destination) == 2.0
    np.testing.assert_array_equal(mapping.link_ids[links], [12])
    np.testing.assert_array_equal(mapping.directions[links], [1])
    np.testing.assert_array_equal(mapping.path_nodes(origin, links), [20, 30])


def test_compact_costs_have_no_removed_link_slot():
    graph = chain_graph()
    context = make_routing_context(graph, compact=True)

    assert context.link_count < graph.num_links
    assert len(context.costs) == graph.compact_num_links
    assert search(context, 0).path_cost_to(1) == 6.0


def test_missing_cost_field_is_rejected():
    graph = diamond()
    graph.cost_field = None

    with pytest.raises(ValueError, match="cost field"):
        make_routing_context(graph)


def test_mapping_copies_ids_and_keeps_context_link_order():
    graph = chain_graph()
    context = make_routing_context(graph)
    node_ids = graph.all_nodes.copy()
    link_ids = graph.graph.link_id.to_numpy(copy=True)
    directions = graph.graph.direction.to_numpy(copy=True)
    mapping = GraphMapping(context, node_ids, link_ids, directions)

    for source, snapshot in (
        (node_ids, mapping.node_ids),
        (link_ids, mapping.link_ids),
        (directions, mapping.directions),
    ):
        assert not np.shares_memory(source, snapshot)
        assert not snapshot.flags.writeable
        source[:] = 0

    origin = mapping.node_index(np.int64(40))
    results = search(context, origin)
    links = results.path_links_to(mapping.node_index(10))

    np.testing.assert_array_equal(mapping.link_ids[links], [55, 12, 71])
    np.testing.assert_array_equal(mapping.directions[links], [-1, -1, -1])
    np.testing.assert_array_equal(mapping.path_nodes(origin, links), [40, 30, 20, 10])
    np.testing.assert_array_equal(mapping.path_nodes(origin, results.path_links_to(origin)), [40])

    with pytest.raises(ValueError, match="not present"):
        mapping.node_index(999)

    with pytest.raises(TypeError, match="boolean"):
        mapping.node_index(True)


@pytest.mark.parametrize(
    "nodes, links, directions, error",
    [
        ([10], [71], [1], ValueError),
        ([10, 10], [71], [1], ValueError),
        ([10.0, 20.0], [71], [1], TypeError),
        ([10, 20], [], [1], ValueError),
        ([10, 20], [71], [0], ValueError),
    ],
)
def test_mapping_checks_array_sizes_and_ids(nodes, links, directions, error):
    context = make_context([0, 1, 1], [1], [1.0])

    with pytest.raises(error):
        GraphMapping(context, nodes, links, directions)


def test_mapping_rejects_graph_objects():
    with pytest.raises(TypeError, match="routing context"):
        GraphMapping(diamond(), [], [], [])
