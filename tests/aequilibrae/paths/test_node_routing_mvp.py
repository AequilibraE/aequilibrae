"""Contracts for contexts, queries and graph-independent search storage."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest

from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.context import NodeBasedContext
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults

from .routing_helpers import allocate_results, assert_state_tree, make_context, search


@pytest.fixture
def context():
    return make_context([0, 2, 4, 5, 5, 5], [1, 2, 2, 3, 3], [1, 4, 1, 10, 1])


def test_results_are_only_path_storage(context):
    results = allocate_results(context)
    assert results.origin is results.root is None
    assert results.settled_count == 0
    assert not results.exhausted
    assert not results.all_targets_reached
    assert not results.reachable_to(0)
    assert results.path_links_to(0).size == 0
    assert np.isinf(results.path_cost_to(0))
    for name in (
        "context",
        "workspace",
        "prepared_skims",
        "destination_mask",
        "path_nodes_to",
        "path_nodes",
        "path_cost",
        "network_loading",
        "skim_fields",
        "select_link_loading",
    ):
        assert not hasattr(results, name)
    for name in ("predecessors", "connectors", "settlement_order", "terminal_states"):
        values = getattr(results, name)
        assert values.dtype == np.uintp
        assert np.all(values == results.sentinel)
        assert not values.flags.writeable
    for name in ("distances", "turn_costs"):
        assert np.all(np.isinf(getattr(results, name)))
    assert not hasattr(context, "make_results")
    assert not hasattr(context, "link_ids")


def test_explicit_context_query_results(context):
    mask = np.array([False, True, True, False, False])
    query = SearchQuery(context.node_count, 0, mask)
    results = allocate_results(context)
    assert dijkstra(context, query, results) is results
    assert results.target_count == results.reached_target_count == 2
    assert results.all_targets_reached and not results.exhausted
    assert results.settled_count == 3
    np.testing.assert_array_equal(results.path_links_to(2), [0, 2])
    assert results.path_cost_to(2) == 2
    assert not results.reachable_to(3)
    np.testing.assert_array_equal(mask, query.target_mask)
    assert_state_tree(context, results)


def test_full_search_and_unreachable_target(context):
    full = search(context, 0)
    assert full.exhausted and full.all_targets_reached
    assert full.target_count == full.reached_target_count == 0
    assert full.settled_count == 4
    assert not full.reachable_to(4)
    assert np.isinf(full.path_cost_to(4))
    targeted = search(context, 0, [1, 4])
    assert targeted.exhausted and not targeted.all_targets_reached
    assert targeted.target_count == 2 and targeted.reached_target_count == 1
    assert_state_tree(context, targeted)


def test_target_stop_does_not_claim_exhaustion_with_empty_heap():
    context = make_context([0, 1, 2, 2], [1, 2], [1, 1])
    results = search(context, 0, 1)
    assert not results.exhausted
    assert not results.reachable_to(2)  # Target 1's outgoing link was not explored.


def test_reuse_updates_metadata_and_retained_views(context):
    results = search(context, 0, 3)
    names = ("predecessors", "connectors", "settlement_order", "terminal_states", "distances", "turn_costs")
    views = [getattr(results, name) for name in names]
    snapshot = results.predecessors.copy()
    query = SearchQuery(context.node_count, 4)
    dijkstra(context, query, results)
    assert results.origin == results.root == 4
    assert results.settled_count == 1 and results.exhausted
    assert np.all(views[0] == results.sentinel)
    assert snapshot[3] == 2
    query.origin = 0
    dijkstra(context, query, results)
    assert results.settled_count == 4
    assert results.path_cost_to(3) == 3
    for name, view in zip(names, views, strict=True):
        assert np.shares_memory(view, getattr(results, name))
    assert_state_tree(context, results)


def test_results_reuse_depends_on_dimensions_not_context_identity(context):
    results = search(context, 0)
    other = context.with_costs(np.ones(context.link_count))
    search(other, 0, results=results)
    assert results.path_cost_to(3) == 2
    assert results.path_turn_cost_to(3) == 0


def test_dimension_checks_happen_before_writes(context):
    results = search(context, 0)
    before = results.distances.copy()
    with pytest.raises(ValueError, match="query node_count"):
        dijkstra(context, SearchQuery(2, 0), results)
    for sizes in [(5, 4, 5), (4, 5, 5), (5, 5, 4)]:
        with pytest.raises(ValueError, match="results dimensions"):
            dijkstra(context, SearchQuery(5, 0), SearchResults(*sizes))
    np.testing.assert_array_equal(before, results.distances)


def test_results_do_not_retain_context_costs_or_query():
    costs = np.array([3.0])
    mask = np.array([False, True])
    costs_ref, mask_ref = weakref.ref(costs), weakref.ref(mask)
    context = NodeBasedContext([0, 1, 1], [1], costs)
    query = SearchQuery(2, 0, mask)
    results = allocate_results(context)
    dijkstra(context, query, results)
    del costs, mask, context, query
    gc.collect()
    assert costs_ref() is mask_ref() is None
    assert results.path_cost_to(1) == 3
    np.testing.assert_array_equal(results.path_links_to(1), [0])
    view = results.predecessors
    owner = weakref.ref(view.base)
    del results
    gc.collect()
    np.testing.assert_array_equal(view, [np.iinfo(np.uintp).max, 0])
    assert owner() is not None
    del view
    gc.collect()
    assert owner() is None


def test_topology_is_copied_costs_are_borrowed_and_rebound():
    fs = np.array([0, 99, 1, 99, 1])[::2]
    heads = np.array([1])
    costs = np.array([3.0])
    context = NodeBasedContext(fs, heads, costs)
    assert not np.shares_memory(fs, context.fs)
    assert not np.shares_memory(heads, context.heads)
    assert np.shares_memory(costs, context.costs)
    assert costs.flags.writeable
    fs[:] = 99
    heads[:] = 99
    costs[0] = 4
    assert search(context, 0).path_cost_to(1) == 4
    new_costs = np.array([7.0])
    new_costs.flags.writeable = False
    other = context.with_costs(new_costs)
    assert np.shares_memory(other.fs, context.fs)
    assert np.shares_memory(other.heads, context.heads)
    assert search(other, 0).path_cost_to(1) == 7
    assert search(context, 0).path_cost_to(1) == 4
    context.update_costs(new_costs)
    assert np.shares_memory(context.costs, new_costs)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [1.0],
        np.ones(2),
        np.array([-1.0]),
        np.array([np.nan]),
        np.ones(1, dtype=np.float32),
        np.ones(4)[::2],
        np.ndarray((1,), dtype=np.float64, buffer=bytearray(9), offset=1),
    ],
)
def test_invalid_cost_update_keeps_previous_binding(bad):
    context = make_context([0, 1, 1], [1], [3])
    before = context.costs
    with pytest.raises((TypeError, ValueError)):
        context.update_costs(bad)
    assert np.shares_memory(before, context.costs)
    assert search(context, 0).path_cost_to(1) == 3


def test_query_borrows_mask_and_has_one_full_search_representation():
    mask = np.array([False, True])
    query = SearchQuery(2, 0, mask)
    assert np.shares_memory(query.target_mask, mask)
    assert mask.flags.writeable
    assert query.target_count == 1
    query.origin = np.uint64(1)
    assert query.origin == 1
    with pytest.raises(ValueError, match="use None"):
        SearchQuery(2, 0, np.zeros(2, dtype=bool))
    for bad in (np.ones(3, dtype=bool), np.ones(4, dtype=bool)[::2], np.ones(2), [False, True]):
        with pytest.raises((TypeError, ValueError)):
            SearchQuery(2, 0, bad)
    for bad in (-1, 2, 2**100, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            query.origin = bad
    assert query.origin == 1


@pytest.mark.parametrize("turn", [False, True])
def test_readonly_views_and_reinitialization(turn):
    context = make_context([0, 1, 1], [1], [3], turn=turn)
    results = search(context, 0)
    query = SearchQuery(2, 0, np.array([False, True]))
    for view in (
        context.fs,
        context.heads,
        context.costs,
        query.target_mask,
        results.predecessors,
        results.distances,
        results.terminal_states,
    ):
        with pytest.raises(ValueError):
            view[0] = 0
        with pytest.raises(ValueError):
            view.flags.writeable = True
    with pytest.raises(RuntimeError):
        context.__init__([0, 1, 1], [1], np.array([2.0]))
    with pytest.raises(RuntimeError):
        results.__init__(2, 2, 1)
    with pytest.raises(RuntimeError):
        query.__init__(2, 0)


@pytest.mark.parametrize("turn", [False, True])
def test_centroid_blocking_is_a_search_input(turn):
    # Nodes 0 and 1 are centroids. A path from 0 may end at 1, but not pass through it.
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 10, 1], turn=turn, blocked_centroid_count=2)
    results = search(context, 0)
    assert results.path_cost_to(1) == 1
    assert results.path_cost_to(2) == 10
    search(context, 1, results=results)
    assert results.path_cost_to(2) == 1


@pytest.mark.parametrize("turn", [False, True])
def test_shared_context_separate_workers(turn):
    n = 500
    context = make_context(np.r_[np.arange(n), n - 1], np.arange(1, n), np.ones(n - 1), turn=turn)

    def run(origin):
        results = allocate_results(context)
        query = SearchQuery(n, origin)
        for _ in range(5):
            dijkstra(context, query, results)
        return results.path_links_to(n - 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(run, range(8)))
    for origin, path in enumerate(paths):
        np.testing.assert_array_equal(path, np.arange(origin, n - 1))


@pytest.mark.parametrize(
    "fs, heads",
    [
        ([], []),
        ([0], []),
        ([[0, 1]], [0]),
        ([0.0, 1.0], [0]),
        ([0, 1], [[0]]),
        ([0, 1], [0.0]),
        ([0, 1], [-1]),
        ([-1, 1], [0]),
        ([1, 1], [0]),
        ([0, 2], [0]),
        ([0, 2, 1], [0]),
        ([0, 1], [1]),
    ],
)
def test_invalid_topology(fs, heads):
    with pytest.raises(ValueError):
        NodeBasedContext(fs, heads, np.ones(len(heads)))


@pytest.mark.parametrize("turn", [False, True])
def test_zero_cost_parallel_links_and_infinite_link(turn):
    context = make_context([0, 4, 5, 5], [0, 1, 1, 2, 2], [0, 2, 0, np.inf, 0], turn=turn)
    results = search(context, 0, 2)
    np.testing.assert_array_equal(results.path_links_to(2), [2, 4])
    assert results.path_cost_to(2) == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize("destination", [-1, 5, 2**100, True, 1.5, None])
def test_invalid_path_query(context, destination):
    results = search(context, 0)
    for operation in (results.reachable_to, results.path_links_to, results.path_cost_to, results.path_turn_cost_to):
        with pytest.raises((ValueError, TypeError)):
            operation(destination)


@pytest.mark.parametrize("seed", [17, 29, 42])
def test_random_multigraph_against_networkx(seed):
    rng = np.random.default_rng(seed)
    n = 12
    edges = [
        (a, b, float(rng.integers(0, 10))) for a in range(n) for b in range(n) for _ in range(2) if rng.random() < 0.08
    ]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = make_context(fs, [b for _, b, _ in edges], [cost for _, _, cost in edges])
    results = allocate_results(context)
    for origin in range(n):
        oracle = nx.single_source_dijkstra_path_length(graph, origin)
        for destination in range(n):
            search(context, origin, destination, results)
            assert_state_tree(context, results)
            assert results.path_cost_to(destination) == oracle.get(destination, np.inf)
            if results.reachable_to(destination):
                links = results.path_links_to(destination)
                assert context.costs[links].sum() == oracle[destination]
                nodes = np.r_[np.array([origin], dtype=np.uintp), context.heads[links]]
                assert nodes[-1] == destination
                assert np.all(links >= context.fs[nodes[:-1]])
                assert np.all(links < context.fs[nodes[:-1] + 1])
