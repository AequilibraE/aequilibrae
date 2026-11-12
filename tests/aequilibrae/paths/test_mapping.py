"""Standalone mapping behaviour, without preparing a routing graph."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths.mapping import GraphMapping, NetworkMapping

BASE = 10**12


@pytest.fixture
def links():
    # The middle physical link opposes the forward chain's traversal direction.
    # The final one-way branch is absent from the compressed graph.
    return pd.DataFrame(
        {
            "link_id": [BASE + 9, BASE + 2, BASE + 7, BASE + 4],
            "a_node": [BASE + 10, BASE + 30, BASE + 30, BASE + 20],
            "b_node": [BASE + 20, BASE + 20, BASE + 40, BASE + 50],
            "direction": [0, 0, 0, 1],
            "modes": ["c", "c", "c", "w"],
        },
        index=[8, 3, 3, 1],
    )


@pytest.fixture
def network(links):
    return NetworkMapping(links)


@pytest.fixture
def mapping(network):
    return GraphMapping(
        network,
        graph_node_to_network=np.array([3, 0]),
        expansion_offsets=np.array([0, 3, 6]),
        expansion_links=np.array([0, 3, 4, 5, 2, 1]),
    )


def test_network_order_and_sparse_ids(network):
    np.testing.assert_array_equal(network.physical_link_ids - BASE, [9, 2, 7, 4])
    np.testing.assert_array_equal(network.link_ids - BASE, [9, 9, 2, 2, 7, 7, 4])
    np.testing.assert_array_equal(network.directions, [1, -1, 1, -1, 1, -1, 1])
    np.testing.assert_array_equal(network.node_ids - BASE, [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(network.link_tails, [0, 1, 2, 1, 2, 3, 1])
    np.testing.assert_array_equal(network.link_heads, [1, 0, 1, 2, 3, 2, 4])
    assert network.node_count == 5
    assert network.link_count == 7
    # Mode metadata does not filter the whole-network mapping.
    assert network.link_index(BASE + 4, 1) == 6
    np.testing.assert_array_equal(network.directed_link_indices, [[0, 1], [2, 3], [4, 5], [6, -1]])
    assert not network.directed_link_indices.flags.writeable


def test_directed_link_indices_mark_missing_directions(links):
    network = NetworkMapping(links.assign(direction=[0, -1, 1, -1]))
    np.testing.assert_array_equal(network.directed_link_indices, [[0, 1], [-1, 2], [3, -1], [-1, 4]])


def test_network_mapping_retains_nodes_without_links(links):
    network = NetworkMapping(links, nodes=np.array([BASE + 99, BASE + 10, BASE + 98]))
    np.testing.assert_array_equal(network.node_ids - BASE, [10, 20, 30, 40, 50, 99, 98])
    np.testing.assert_array_equal(network.node_index([BASE + 99, BASE + 98]), [5, 6])


def test_node_order_is_first_appearance(links):
    network = NetworkMapping(links.iloc[[2, 0, 3, 1]])
    np.testing.assert_array_equal(network.node_ids - BASE, [30, 40, 10, 20, 50])
    np.testing.assert_array_equal(network.physical_link_ids - BASE, [7, 9, 4, 2])


@pytest.mark.parametrize("scalar", [int, np.int64, np.uint64, np.asarray])
def test_scalar_queries(mapping, scalar):
    assert mapping.graph_link(scalar(BASE + 9), 1) == 0
    assert type(mapping.graph_link(scalar(BASE + 9), 1)) is int
    assert mapping.graph_link(scalar(BASE + 2), 1) == 1
    assert mapping.graph_node(scalar(BASE + 10)) == 1
    assert type(mapping.graph_node(scalar(BASE + 10))) is int
    assert mapping.network_node(scalar(0)) == BASE + 40
    assert type(mapping.network_node(scalar(0))) is int
    assert type(mapping.network.node_index(scalar(BASE + 10))) is int
    np.testing.assert_array_equal(mapping.expand_link_indices(scalar(0)), [0, 3, 4])


def test_array_queries_and_broadcasting(mapping):
    np.testing.assert_array_equal(mapping.graph_link([BASE + 9, BASE + 2, BASE + 7], [1, -1, 1]), [0, 0, 0])
    np.testing.assert_array_equal(mapping.graph_link(BASE + 2, [1, -1, 1]), [1, 0, 1])
    np.testing.assert_array_equal(mapping.graph_link([BASE + 7, BASE + 9], -1), [1, 1])
    np.testing.assert_array_equal(mapping.graph_node([BASE + 10, BASE + 40, BASE + 10]), [1, 0, 1])
    np.testing.assert_array_equal(mapping.network_node([1, 0, 1]), [BASE + 10, BASE + 40, BASE + 10])
    assert mapping.graph_node([BASE + 10]).dtype == np.dtype(np.intp)


def test_noncontiguous_readonly_queries(mapping):
    ids = np.array([BASE + 9, 0, BASE + 7, 0], dtype=np.int64)[::2]
    ids.flags.writeable = False
    np.testing.assert_array_equal(mapping.graph_link(ids, 1), [0, 0])
    graph_links = np.array([1, 9, 0, 9])[::2]
    graph_links.flags.writeable = False
    np.testing.assert_array_equal(mapping.expand_links(graph_links)[0] - BASE, [7, 2, 9, 9, 2, 7])
    np.testing.assert_array_equal(mapping.expand_link_indices(graph_links), [5, 2, 1, 0, 3, 4])


@pytest.mark.parametrize("link,direction", [(99, 1), (4, 1), (4, -1)])
def test_missing_links_raise_for_whole_query(mapping, link, direction):
    with pytest.raises(KeyError):
        mapping.graph_link(BASE + link, direction)
    with pytest.raises(KeyError):
        mapping.graph_link([BASE + 9, BASE + link], [1, direction])
    assert mapping.graph_link(BASE + 9, 1) == 0


@pytest.mark.parametrize("node", [20, 30, 50, 99])
def test_missing_nodes_raise(mapping, node):
    with pytest.raises(KeyError):
        mapping.graph_node(BASE + node)
    with pytest.raises(KeyError):
        mapping.graph_node([BASE + 10, BASE + node])


@pytest.mark.parametrize("direction", [0, 2, True, 1.0, "1", [[1]]])
def test_invalid_lookup_direction(mapping, direction):
    with pytest.raises(ValueError):
        mapping.graph_link(BASE + 9, direction)


@pytest.mark.parametrize("value", [-1, 1.5, True, "10", [[10]], None])
def test_invalid_external_ids(mapping, value):
    with pytest.raises(ValueError):
        mapping.graph_node(value)
    with pytest.raises(ValueError):
        mapping.graph_link(value, 1)


@pytest.mark.parametrize("value", [-1, 2, 2**63, 0.5, True, [[0]]])
def test_invalid_graph_indices(mapping, value):
    with pytest.raises(ValueError):
        mapping.expand_link_indices(value)
    with pytest.raises(ValueError):
        mapping.expand_links(value)
    with pytest.raises(ValueError):
        mapping.network_node(value)


def test_broadcast_mismatch(mapping):
    with pytest.raises(ValueError):
        mapping.graph_link([BASE + 9, BASE + 2], [1, -1, 1])


@pytest.mark.parametrize(
    "graph_links,expected",
    [(0, [0, 3, 4]), ([1, 0, 1], [5, 2, 1, 0, 3, 4, 5, 2, 1]), ([], [])],
)
def test_expand_link_indices(mapping, graph_links, expected):
    indices = mapping.expand_link_indices(graph_links)
    np.testing.assert_array_equal(indices, expected)
    assert indices.dtype == np.dtype(np.intp)
    assert indices.flags.c_contiguous
    assert indices.flags.writeable
    assert not np.shares_memory(indices, mapping.expansion_links)
    ids, directions = mapping.expand_links(graph_links)
    np.testing.assert_array_equal(mapping.network.link_ids[indices], ids)
    np.testing.assert_array_equal(mapping.network.directions[indices], directions)


def test_expand_link_indices_gathers_attributes(mapping):
    indices = mapping.expand_link_indices([1, 0])
    costs = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0])
    labels = np.array(["a", "b", "c", "d", "e", "f", "g"])
    attributes = np.column_stack((costs, costs * 2))
    np.testing.assert_array_equal(costs[indices], [13, 5, 3, 2, 7, 11])
    np.testing.assert_array_equal(labels[indices], ["f", "c", "b", "a", "d", "e"])
    np.testing.assert_array_equal(attributes[indices], [[13, 26], [5, 10], [3, 6], [2, 4], [7, 14], [11, 22]])


def test_ordered_expansion(mapping):
    ids, directions = mapping.expand_links(0)
    np.testing.assert_array_equal(ids - BASE, [9, 2, 7])
    np.testing.assert_array_equal(directions, [1, -1, 1])
    ids, directions = mapping.expand_links([1, 0, 1])
    np.testing.assert_array_equal(ids - BASE, [7, 2, 9, 9, 2, 7, 7, 2, 9])
    np.testing.assert_array_equal(directions, [-1, 1, -1, 1, -1, 1, -1, 1, -1])
    assert ids.dtype == np.dtype(np.uint64)
    assert directions.dtype == np.dtype(np.int8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [None, 0, 1, 3])
@pytest.mark.parametrize("use_out", [False, True])
def test_projection_preserves_dtype_and_shape(mapping, dtype, classes, use_out):
    loads = np.array([7, 11], dtype=dtype)
    expected = np.array([7, 11, 11, 7, 7, 11, 0], dtype=dtype)
    if classes is not None:
        loads = np.repeat(loads[:, None], classes, axis=1)
        expected = np.repeat(expected[:, None], classes, axis=1)
    loads.flags.writeable = False
    out = np.full(expected.shape, 99, dtype=loads.dtype) if use_out else None
    result = mapping.project_loads(loads, out=out)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == loads.dtype
    assert not np.shares_memory(result, loads)
    assert result.flags.writeable
    if use_out:
        assert result is out


def test_projection_noncontiguous_and_nonfinite(mapping):
    loads = np.array([[7.0, 1, 14, 2], [11, 3, 22, 4]])[:, ::2]
    expected = np.array([7, 11, 11, 7, 7, 11, 0])
    np.testing.assert_array_equal(mapping.project_loads(loads), np.column_stack((expected, expected * 2)))
    projected = mapping.project_loads(np.array([np.nan, np.inf]))
    assert np.isnan(projected[0])
    assert np.isinf(projected[1])
    assert projected[-1] == 0


def test_empty_queries(mapping):
    for result in (
        mapping.graph_link([], 1),
        mapping.graph_node([]),
        mapping.network_node([]),
        mapping.expand_link_indices([]),
        *mapping.expand_links([]),
    ):
        assert result.shape == (0,)


def test_inverse_arrays_and_retained_isolated_nodes(network, mapping):
    np.testing.assert_array_equal(mapping.network_link_to_graph, [0, 1, 1, 0, 0, 1, -1])
    np.testing.assert_array_equal(mapping.network_node_to_graph, [1, -1, -1, 0, -1])
    isolated = GraphMapping(network, [4, 0], [0], [])
    assert isolated.link_count == 0
    assert isolated.node_count == 2
    assert isolated.graph_node(BASE + 50) == 0
    np.testing.assert_array_equal(isolated.project_loads(np.empty(0)), np.zeros(7))
    np.testing.assert_array_equal(isolated.project_loads(np.empty((0, 2))), np.zeros((7, 2)))
    with pytest.raises(KeyError):
        isolated.graph_link(BASE + 9, 1)


def test_identity_mapping_shares_network(network, mapping):
    identity = GraphMapping(network, np.arange(5), np.arange(8), np.arange(7))
    assert identity.network is mapping.network
    ids, directions = identity.expand_links(np.arange(7))
    np.testing.assert_array_equal(ids, network.link_ids)
    np.testing.assert_array_equal(directions, network.directions)
    np.testing.assert_array_equal(identity.project_loads(np.arange(7, dtype=np.float64)), np.arange(7))
    np.testing.assert_array_equal(identity.expand_link_indices(np.arange(7)), np.arange(7))


def test_snapshots_are_readonly_and_independent(links, network):
    nodes, offsets, expansion = np.array([3, 0]), np.array([0, 3, 6]), np.array([0, 3, 4, 5, 2, 1])
    mapping = GraphMapping(network, nodes, offsets, expansion)
    nodes[:], offsets[:], expansion[:] = 0, 0, 0
    links.loc[:, "link_id"] = 100
    links.loc[:, "a_node"] = 200
    assert mapping.graph_link(BASE + 9, 1) == 0
    assert mapping.network_node(0) == BASE + 40
    for owner in (network, mapping):
        for values in vars(owner).values():
            if isinstance(values, np.ndarray):
                assert values.flags.c_contiguous
                assert not values.flags.writeable
    with pytest.raises(FrozenInstanceError):
        mapping.network = network
    with pytest.raises(ValueError):
        mapping.expansion_links[0] = 1
    result, _ = mapping.expand_links(0)
    result[:] = 0
    assert mapping.expand_links(0)[0][0] == BASE + 9
    indices = mapping.expand_link_indices(0)
    indices[:] = 0
    np.testing.assert_array_equal(mapping.expand_link_indices(0), [0, 3, 4])


def test_empty_network():
    network = NetworkMapping(pd.DataFrame(columns=["link_id", "a_node", "b_node", "direction"]))
    mapping = GraphMapping(network, [], [0], [])
    assert network.node_count == network.link_count == mapping.node_count == mapping.link_count == 0
    assert mapping.project_loads(np.empty((0, 3))).shape == (0, 3)
    assert mapping.expand_link_indices([]).shape == (0,)
    with pytest.raises(ValueError):
        mapping.expand_link_indices(0)
    with pytest.raises(KeyError):
        mapping.graph_node(0)


def test_uint64_id_range():
    maximum = np.iinfo(np.uint64).max
    links = pd.DataFrame(
        {
            "link_id": np.array([maximum], dtype=np.uint64),
            "a_node": np.array([0], dtype=np.uint64),
            "b_node": np.array([maximum], dtype=np.uint64),
            "direction": [-1],
        }
    )
    network = NetworkMapping(links)
    mapping = GraphMapping(network, [1, 0], [0, 1], [0])
    assert mapping.graph_link(maximum, -1) == 0
    assert mapping.network_node(0) == maximum
    assert mapping.graph_node(maximum) == 0
    assert mapping.expand_links(0)[0][0] == maximum


@pytest.mark.parametrize(
    "column,values,match",
    [
        ("link_id", [1, 1, 2, 3], "unique"),
        ("link_id", [1.0, 2.0, 3.0, 4.0], "integers"),
        ("a_node", [-1, 1, 2, 3], "nonnegative"),
        ("direction", [0, 0, 0, 2], "directions"),
        ("direction", [False, True, True, False], "integers"),
    ],
)
def test_invalid_network_columns(links, column, values, match):
    links[column] = values
    with pytest.raises(ValueError, match=match):
        NetworkMapping(links)


def test_missing_network_column(links):
    with pytest.raises(ValueError, match="required columns"):
        NetworkMapping(links.drop(columns="a_node"))


@pytest.mark.parametrize(
    "nodes,offsets,expansion,match",
    [
        ([3, 0], [], [], "offsets"),
        ([3, 0], [1, 3], [0, 3, 4], "offsets"),
        ([3, 0], [0, 2], [0, 3, 4], "offsets"),
        ([3, 0], [0, 0, 3], [0, 3, 4], "offsets"),
        ([3, 0], [0, 3, 2, 6], [0, 3, 4, 5, 2, 1], "offsets"),
        ([3, 0], [0, 4], [0, 3, 4], "expansion_offsets"),
        ([3, 0], [0, 3], [0, 3, 7], "expansion_links"),
        ([3, 0], [0, 3], [0, 3, -1], "expansion_links"),
        ([3, 0], [0, 2], [0, 0], "at most one"),
        ([3, 0, 3], [0, 3], [0, 3, 4], "unique"),
        ([3, 0, 5], [0, 3], [0, 3, 4], "graph_node_to_network"),
        ([3], [0, 3], [0, 3, 4], "endpoints"),
        ([3, 0], [0, 3], [0, 2, 4], "connected chain"),
        ([[3, 0]], [0, 3], [0, 3, 4], "one-dimensional"),
        ([3, 0], [0, 3], [[0, 3, 4]], "one-dimensional"),
        ([3, 0], [[0, 3]], [0, 3, 4], "one-dimensional"),
        (0, [0], [], "one-dimensional"),
        ([3, 0], [0, 1], 0, "one-dimensional"),
    ],
)
def test_invalid_compression_arrays(network, nodes, offsets, expansion, match):
    with pytest.raises(ValueError, match=match):
        GraphMapping(network, nodes, offsets, expansion)
