"""Shared routing setup, snapshots and external IDs."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Network
from aequilibrae.paths.routing_context import GraphMapping

from .routing_helpers import diamond_graph, make_context, search


def chain_network(centroids=True):
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 55],
            "a_node": [10, 20, 30],
            "b_node": [20, 30, 40],
            "direction": [0, 0, 0],
            "time": [1.0, 2.0, 3.0],
        }
    )
    return Network(links, centroids=[10, 40] if centroids else None)


def compressed_uturn_graph(with_return_chain=True, *, compact=True):
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
    links = pd.DataFrame(
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
        links = links.loc[~links.link_id.isin([4, 5])].copy()

    turns = pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [6], "penalty": [np.inf]})
    return Network(links, centroids=[1, 3, 6], turn_restrictions=turns).build_graph(
        "c", "time", compact=compact, block_centroid_flows=False
    )


@pytest.mark.parametrize("compact", [False, True])
def test_compact_dijkstra_allows_return_through_another_chain(compact):
    graph = compressed_uturn_graph(compact=compact)
    if compact:
        assert 4 not in graph.all_nodes
        assert 5 not in graph.all_nodes

    expected_nodes = [1, 2, 3, 2, 6] if compact else [1, 2, 4, 3, 5, 2, 6]
    np.testing.assert_array_equal(graph.compute_path(1, 6).path_nodes, expected_nodes)

    # Test the routing context directly (internal infrastructure test).
    origin = graph.mapping.graph_node(1)
    destination = graph.mapping.graph_node(6)
    results = search(graph.routing, origin, destination)

    # Links 2/3 form the outward chain and links 4/5 are the return via 5.
    links, directions = graph.mapping.expand_links(results.path_links_to(destination))
    np.testing.assert_array_equal(links, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(directions, 1)
    assert results.path_cost_to(destination) == 6.0


@pytest.mark.parametrize("compact", [False, True])
def test_compact_dijkstra_bans_reversal_through_same_chain(compact):
    graph = compressed_uturn_graph(with_return_chain=False, compact=compact)

    # Without 3 -> 5 -> 2, the only return is 3 -> 4 -> 3, so 6 is unreachable.
    assert graph.compute_path(1, 6).path is None

    # Check the routing context as well.
    origin = graph.mapping.graph_node(1)
    destination = graph.mapping.graph_node(6)
    results = search(graph.routing, origin, destination)
    assert not results.reachable_to(destination)


@pytest.mark.parametrize("centroids", [False, True])
def test_full_graph_keeps_intermediate_nodes(centroids):
    graph = chain_network(centroids).build_graph("c", "time", block_centroid_flows=False)
    origin, destination = graph.mapping.graph_node([20, 30])
    results = search(graph.routing, origin)
    links, directions = graph.mapping.expand_links(results.path_links_to(destination))

    assert graph.num_nodes == 4
    assert results.path_cost_to(destination) == 2.0
    np.testing.assert_array_equal(links, [12])
    np.testing.assert_array_equal(directions, [1])
    np.testing.assert_array_equal(graph.compute_path(20, 30).path_nodes, [20, 30])


def test_compact_costs_have_no_removed_link_slot():
    network = chain_network()
    graph = network.build_graph("c", "time", compact=True, block_centroid_flows=False)
    assert graph.num_links < network.mapping.link_count
    assert len(graph.cost) == graph.num_links
    origin, destination = graph.mapping.graph_node([10, 40])
    assert search(graph.routing, origin).path_cost_to(destination) == 6.0


def test_missing_cost_field_is_rejected():
    with pytest.raises(ValueError, match="not in network"):
        chain_network().build_graph("c", "missing")


def test_mapping_copies_ids_and_keeps_context_link_order():
    graph = chain_network().build_graph("c", "time", block_centroid_flows=False)
    context = graph.routing
    node_ids = graph.all_nodes.copy()
    link_ids, directions = graph.mapping.expand_links(np.arange(graph.num_links))
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
        GraphMapping(diamond_graph(), [], [], [])
