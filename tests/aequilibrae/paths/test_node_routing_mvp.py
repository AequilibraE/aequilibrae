"""Standalone MVP tests: deliberately do not construct a production Graph."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest
from aequilibrae.paths.cython.graph_context import NodeBasedContext
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.search_results import SearchResults


@pytest.fixture
def context():
    # 0 -> 1 (1), 0 -> 2 (4), 1 -> 2 (1), 1 -> 3 (10), 2 -> 3 (1)
    # Node 4 is isolated. The best 0 -> 3 route requires a decrease-key at 2 and 3.
    return NodeBasedContext([0, 2, 4, 5, 5, 5], [1, 2, 2, 3, 3], [1, 4, 1, 10, 1])


def test_initialized_results(context):
    results = context.make_results()
    assert isinstance(results, SearchResults)
    assert results.context is context
    assert results.origin is None
    assert results.destination is None
    assert results.settled_count == 0
    assert not results.reachable
    assert results.path_nodes.size == results.path_links.size == 0
    assert results.sentinel == np.iinfo(np.uintp).max
    for array in (results.predecessors, results.connectors, results.reached_first):
        assert array.dtype == np.dtype(np.uintp)
        assert array.shape == (context.node_count,)
        assert np.all(array == results.sentinel)
        assert not array.flags.writeable
        assert not array.flags.owndata


def test_point_to_point(context):
    results = context.make_results()
    assert dijkstra(context, 0, 3, results) is results
    assert results.origin == 0
    assert results.destination == 3
    assert results.reachable
    assert results.settled_count == 4
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2, 3])
    np.testing.assert_array_equal(results.path_links, [0, 2, 4])
    np.testing.assert_array_equal(results.predecessors, [results.sentinel, 0, 1, 2, results.sentinel])
    np.testing.assert_array_equal(results.connectors, [results.sentinel, 0, 2, 4, results.sentinel])
    np.testing.assert_array_equal(results.reached_first, [0, 1, 2, 3, results.sentinel])
    assert context.costs[results.path_links].sum() == 3


def test_multiple_destinations_and_target_specific_paths(context):
    results = dijkstra(context, 0, [1, 2])
    assert results.destination is None
    assert results.destination_count == results.reached_destination_count == 2
    assert results.reachable
    np.testing.assert_array_equal(results.destinations, [1, 2])
    np.testing.assert_array_equal(results.destination_mask, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(results.path_nodes_to(1), [0, 1])
    np.testing.assert_array_equal(results.path_nodes_to(2), [0, 1, 2])
    assert results.path_cost_to(1) == 1
    assert results.path_cost_to(2) == 2
    assert results.settled_count == 3
    assert results.predecessors[3] == results.sentinel
    with pytest.raises(ValueError, match="multi-target"):
        _ = results.path_nodes


def test_destination_mask_and_unreachable_target(context):
    results = dijkstra(context, 0, np.array([False, True, False, False, True]))
    assert not results.reachable
    assert results.destination_count == 2
    assert results.reached_destination_count == 1
    assert results.reachable_to(1)
    assert not results.reachable_to(4)
    assert np.isinf(results.path_cost_to(4))
    np.testing.assert_array_equal(results.path_links_to(4), [])
    with pytest.raises(ValueError, match="one entry per node"):
        dijkstra(context, 0, np.array([True, False]))


@pytest.mark.parametrize("destinations", [[], np.zeros(5, dtype=bool)])
def test_empty_destination_mask_disables_early_exit(context, destinations):
    results = dijkstra(context, 0, destinations)
    assert results.destination_count == results.reached_destination_count == 0
    assert results.all_destinations_reached
    assert results.settled_count == 4
    np.testing.assert_array_equal(results.path_nodes_to(3), [0, 1, 2, 3])


def test_none_requests_all_nodes(context):
    results = dijkstra(context, 0, None)
    assert results.destination_count == context.node_count
    assert results.reached_destination_count == 4
    assert not results.all_destinations_reached  # Node 4 is isolated.
    assert results.settled_count == 4
    np.testing.assert_array_equal(results.path_nodes_to(3), [0, 1, 2, 3])


def test_early_exit_clears_tentative_paths(context):
    results = dijkstra(context, 0, 1)
    assert results.settled_count == 2
    np.testing.assert_array_equal(results.path_nodes, [0, 1])
    # Node 2 was inserted by the origin, but was not settled before early exit.
    assert results.predecessors[2] == results.sentinel
    assert results.connectors[2] == results.sentinel
    assert np.all(results.reached_first[2:] == results.sentinel)


def test_unreachable_destination(context):
    results = dijkstra(context, 0, 4)
    assert not results.reachable
    assert results.settled_count == 4
    assert results.path_nodes.size == results.path_links.size == 0
    assert results.predecessors[4] == results.sentinel
    assert results.connectors[4] == results.sentinel


def test_origin_is_destination(context):
    results = dijkstra(context, 2, 2)
    assert results.reachable
    assert results.settled_count == 1
    np.testing.assert_array_equal(results.path_nodes, [2])
    assert results.path_links.size == 0
    assert np.all(results.predecessors == results.sentinel)
    assert np.all(results.connectors == results.sentinel)


@pytest.mark.parametrize("nodes", [1, 3])
def test_edgeless_graph(nodes):
    context = NodeBasedContext(np.zeros(nodes + 1, dtype=np.int64), [], [])
    results = dijkstra(context, 0, nodes - 1)
    assert results.settled_count == 1
    assert results.reachable == (nodes == 1)


def test_zero_cost_parallel_arcs_and_infinite_cost():
    context = NodeBasedContext([0, 4, 5, 5], [0, 1, 1, 2, 2], [0, 2, 0, np.inf, 0])
    results = dijkstra(context, 0, 2)
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2])
    np.testing.assert_array_equal(results.path_links, [2, 4])
    blocked = NodeBasedContext([0, 1, 1], [1], [np.inf])
    assert not dijkstra(blocked, 0, 1).reachable


def test_reuse_preserves_allocations_and_resets_search(context):
    results = dijkstra(context, 0, 3)
    arrays = (results.predecessors, results.connectors, results.reached_first)
    pointers = [array.ctypes.data for array in arrays]
    saved_predecessors = results.predecessors.copy()

    dijkstra(context, 4, 0, results)
    assert not results.reachable
    assert results.origin == 4
    assert results.destination == 0
    assert results.settled_count == 1
    assert np.all(arrays[0] == results.sentinel)
    assert np.all(arrays[1] == results.sentinel)
    assert arrays[2][0] == 4
    assert np.all(arrays[2][1:] == results.sentinel)
    assert saved_predecessors[3] == 2

    # Changing destinations repeatedly must not leave old destination-mask bits.
    dijkstra(context, 0, 1, results)
    dijkstra(context, 0, 3, results)
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2, 3])
    assert pointers == [
        array.ctypes.data for array in (results.predecessors, results.connectors, results.reached_first)
    ]
    for old, new in zip(arrays, (results.predecessors, results.connectors, results.reached_first), strict=True):
        assert np.shares_memory(old, new)


def test_views_survive_wrappers_and_allocations_are_eventually_freed():
    context = NodeBasedContext([0, 1, 1], [1], [3])
    results = dijkstra(context, 0, 1)
    pred_view = results.predecessors
    costs_view = context.costs
    pred_owner = weakref.ref(pred_view.base)
    cost_owner = weakref.ref(costs_view.base)
    del results, context
    gc.collect()
    np.testing.assert_array_equal(pred_view, [np.iinfo(np.uintp).max, 0])
    np.testing.assert_array_equal(costs_view, [3])
    assert pred_owner() is not None
    assert cost_owner() is not None
    del pred_view, costs_view
    gc.collect()
    assert pred_owner() is None
    assert cost_owner() is None


def test_results_pin_context():
    context = NodeBasedContext([0, 1, 1], [1], [3])
    results = context.make_results()
    del context
    gc.collect()
    dijkstra(results.context, 0, 1, results)
    np.testing.assert_array_equal(results.path_nodes, [0, 1])


def test_context_copies_inputs_and_accepts_strided_buffers():
    fs = np.array([0, 99, 1, 99, 1, 99])[::2]
    heads = np.array([1, 99])[::2]
    costs = np.array([3.0, 99])[::2]
    context = NodeBasedContext(fs, heads, costs)
    for source, view in ((fs, context.fs), (heads, context.heads), (costs, context.costs)):
        assert not np.shares_memory(source, view)
        assert view.flags.c_contiguous
        source[:] = 99
    np.testing.assert_array_equal(dijkstra(context, 0, 1).path_nodes, [0, 1])
    assert context.costs[0] == 3
    assert context.link_count == 1
    assert context.node_count == 2


def test_views_reject_writes(context):
    results = dijkstra(context, 0, 3)
    for array in (
        context.fs,
        context.heads,
        context.costs,
        context.link_ids,
        results.predecessors,
        results.connectors,
        results.reached_first,
        results.destination_mask,
    ):
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize(
    "fs, heads, costs",
    [
        ([], [], []),
        ([0], [], []),
        ([[0, 1]], [0], [1]),
        ([0.0, 1.0], [0], [1]),
        ([0, 1], [[0]], [1]),
        ([0, 1], [0.0], [1]),
        ([0, 1], [-1], [1]),
        ([-1, 1], [0], [1]),
        ([1, 1], [0], [1]),
        ([0, 2], [0], [1]),
        ([0, 2, 1], [0], [1]),
        ([0, 1], [1], [1]),
        ([0, 1], [0], []),
        ([0, 1], [0], [[1]]),
        ([0, 1], [0], [-1]),
        ([0, 1], [0], [-np.inf]),
        ([0, 1], [0], [np.nan]),
    ],
)
def test_invalid_graphs(fs, heads, costs):
    with pytest.raises(ValueError):
        NodeBasedContext(fs, heads, costs)


@pytest.mark.parametrize("origin, destination", [(-1, 0), (0, -1), (5, 0), (0, 5), (2**100, 0)])
def test_invalid_node_indices(context, origin, destination):
    with pytest.raises(ValueError, match="node range"):
        dijkstra(context, origin, destination)


@pytest.mark.parametrize("value", [1.5, "1", None])
def test_non_integer_node_index(context, value):
    with pytest.raises(TypeError):
        dijkstra(context, value, 0)


def test_numpy_integer_node_indices(context):
    assert dijkstra(context, np.int64(0), np.uint64(3)).reachable


def test_wrong_context(context):
    other = NodeBasedContext(context.fs, context.heads, context.costs)
    results = dijkstra(context, 0, 3)
    with pytest.raises(ValueError, match="different context"):
        dijkstra(other, 0, 3, results)
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2, 3])


def test_separate_results_can_share_context_across_threads():
    n = 2000
    context = NodeBasedContext(np.r_[np.arange(n), n - 1], np.arange(1, n), np.ones(n - 1))
    results = [context.make_results() for _ in range(8)]

    def run(item):
        origin, result = item
        for _ in range(10):
            dijkstra(context, origin, n - 1, result)
        return result.path_nodes

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(run, enumerate(results)))
    for origin, path in enumerate(paths):
        np.testing.assert_array_equal(path, np.arange(origin, n))


def test_random_graph_against_networkx():
    rng = np.random.default_rng(246)
    n = 20
    edges = [(a, b, float(rng.integers(0, 20))) for a in range(n) for b in range(n) if a != b and rng.random() < 0.12]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = NodeBasedContext(fs, [b for _, b, _ in edges], [cost for _, _, cost in edges])
    results = context.make_results()
    for origin in range(n):
        distances = nx.single_source_dijkstra_path_length(graph, origin)
        for destination in range(n):
            dijkstra(context, origin, destination, results)
            assert results.reachable == (destination in distances)
            if not results.reachable:
                continue
            nodes = results.path_nodes
            links = results.path_links
            assert nodes[0] == origin
            assert nodes[-1] == destination
            assert len(links) == len(nodes) - 1
            np.testing.assert_array_equal(context.heads[links], nodes[1:])
            assert np.all(links >= context.fs[nodes[:-1]])
            assert np.all(links < context.fs[nodes[:-1] + 1])
            assert context.costs[links].sum() == distances[destination]
