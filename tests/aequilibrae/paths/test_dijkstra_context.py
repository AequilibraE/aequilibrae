"""Contracts for contexts, queries and graph-independent search storage."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest

from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.workspaces import SearchWorkspace

from .routing_helpers import allocate_results, assert_state_tree, history_context, make_context, search


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
    workspace = SearchWorkspace(context.node_count, context.state_count)
    assert dijkstra(context, query, results, workspace) is results
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
    workspace = SearchWorkspace(context.node_count, context.state_count)
    dijkstra(context, query, results, workspace)
    assert results.origin == results.root == 4
    assert results.settled_count == 1 and results.exhausted
    assert np.all(views[0] == results.sentinel)
    assert snapshot[3] == 2
    query.origin = 0
    dijkstra(context, query, results, workspace)
    assert results.settled_count == 4
    assert results.path_cost_to(3) == 3
    for name, view in zip(names, views, strict=True):
        assert np.shares_memory(view, getattr(results, name))
    assert_state_tree(context, results)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("targets", [None, [2], [2, 3]])
def test_reset_clears_results_in_place(turn, targets):
    turns = {(0, 1): 0.5} if turn else None
    context = make_context([0, 1, 2, 2, 2], [1, 2], [1, 2], turns)
    results = search(context, 0, targets)
    index_names = ("predecessors", "connectors", "settlement_order", "terminal_states")
    label_names = ("distances", "turn_costs")
    views = {name: getattr(results, name) for name in index_names + label_names}
    assert results.path_cost_to(2) == (3.5 if turn else 3.0)

    for _ in range(2):
        results.reset()
        assert results.origin is results.root is None
        assert results.settled_count == results.target_count == results.reached_target_count == 0
        assert not results.exhausted
        assert not results.all_targets_reached
        for name, view in views.items():
            assert np.shares_memory(view, getattr(results, name))
            assert not view.flags.writeable
            expected = results.sentinel if name in index_names else np.inf
            np.testing.assert_array_equal(view, expected)
        for node in range(context.node_count):
            assert not results.reachable_to(node)
            assert results.path_links_to(node).size == 0
            assert results.path_cost_to(node) == results.path_turn_cost_to(node) == np.inf

    search(context, 1, 2, results)
    assert results.origin == 1
    assert results.root == (context.link_count if turn else 1)
    assert results.target_count == results.reached_target_count == 1
    assert results.all_targets_reached and not results.exhausted
    assert results.path_cost_to(2) == 2.0
    assert results.path_turn_cost_to(2) == 0.0
    for name, view in views.items():
        assert np.shares_memory(view, getattr(results, name))
    assert_state_tree(context, results)


@pytest.mark.parametrize("turn", [False, True])
def test_reset_fresh_and_edgeless_results(turn):
    context = make_context([0, 0], [], [], turn=turn)
    results = allocate_results(context)
    results.reset()
    assert results.origin is results.root is None
    search(context, 0, results=results)
    assert results.path_cost_to(0) == 0.0
    results.reset()
    assert results.origin is results.root is None
    assert results.settled_count == 0
    assert not results.exhausted
    assert not results.reachable_to(0)
    assert results.path_cost_to(0) == np.inf


@pytest.mark.parametrize("turn", [False, True])
def test_path_states_include_root_and_preserve_path_order(turn):
    context = make_context([0, 1, 2, 2, 2], [1, 2], [1, 2], turn=turn)
    results = search(context, 0)
    states = results.path_states_to(2)
    expected = [2, 0, 1] if turn else [0, 1, 2]
    assert states.dtype == np.uintp
    np.testing.assert_array_equal(states, expected)
    np.testing.assert_array_equal(results.connectors[states[1:]], results.path_links_to(2))
    np.testing.assert_array_equal(results.distances[states], [0, 1, 3])
    np.testing.assert_array_equal(results.path_states_to(0), [results.root])
    assert results.path_links_to(0).size == 0
    assert results.path_states_to(3).size == 0

    # Returned paths own their storage and survive another search or reset.
    search(context, 1, results=results)
    np.testing.assert_array_equal(states, expected)
    results.reset()
    np.testing.assert_array_equal(states, expected)
    assert results.path_states_to(0).size == 0


@pytest.mark.parametrize("turn", [False, True])
def test_path_states_require_a_finalized_destination(turn):
    context = make_context([0, 1, 2, 2], [1, 2], [1, 2], turn=turn)
    results = allocate_results(context)
    assert results.path_states_to(0).size == 0
    search(context, 0, 1, results)
    assert not results.exhausted
    assert results.path_states_to(2).size == 0
    assert results.path_states_to(2).dtype == np.uintp
    assert results.path_links_to(2).size == 0


def test_results_reuse_depends_on_dimensions_not_context_identity(context):
    results = search(context, 0)
    other = context.with_costs(np.ones(context.link_count))
    search(other, 0, results=results)
    assert results.path_cost_to(3) == 2
    assert results.path_turn_cost_to(3) == 0


def test_dimension_checks_happen_before_writes(context):
    results = search(context, 0)
    before = results.distances.copy()
    workspace = SearchWorkspace(context.node_count, context.state_count)
    with pytest.raises(ValueError, match="query node_count"):
        dijkstra(context, SearchQuery(2, 0), results, workspace)
    for sizes in [(5, 4, 5), (4, 5, 5), (5, 5, 4)]:
        with pytest.raises(ValueError, match="results dimensions"):
            dijkstra(context, SearchQuery(5, 0), SearchResults(*sizes), workspace)
    np.testing.assert_array_equal(before, results.distances)


def test_results_do_not_retain_context_costs_or_query():
    costs = np.array([3.0])
    mask = np.array([False, True])
    costs_ref, mask_ref = weakref.ref(costs), weakref.ref(mask)
    context = NodeBasedContext([0, 1, 1], [1], costs)
    query = SearchQuery(2, 0, mask)
    results = allocate_results(context)
    workspace = SearchWorkspace(context.node_count, context.state_count)
    dijkstra(context, query, results, workspace)
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
        workspace = SearchWorkspace(context.node_count, context.state_count)
        query = SearchQuery(n, origin)
        for _ in range(5):
            dijkstra(context, query, results, workspace)
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
    for operation in (
        results.reachable_to,
        results.path_states_to,
        results.path_links_to,
        results.path_cost_to,
        results.path_turn_cost_to,
    ):
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


@pytest.mark.parametrize("penalty", [10, np.inf])
def test_nonterminal_arrival_history(penalty):
    context = history_context(penalty)
    results = search(context, 0, [1, 3])
    assert results.root == context.link_count
    assert results.terminal_states[1] == 0
    assert results.predecessors[results.terminal_states[3]] == 3
    np.testing.assert_array_equal(results.path_links_to(1), [0])
    np.testing.assert_array_equal(results.path_links_to(3), [1, 3, 2])
    states = results.path_states_to(3)
    np.testing.assert_array_equal(states, [results.root, 1, 3, 2])
    np.testing.assert_array_equal(results.distances[states], [0, 1, 2, 3])
    np.testing.assert_array_equal(results.path_states_to(1), [results.root, 0])
    assert results.path_cost_to(3) == 3
    assert results.path_turn_cost_to(3) == 0
    assert results.reached_target_count == 2
    assert_state_tree(context, results)


def test_first_link_pays_no_turn_cost_and_results_can_change_mode():
    turn = make_context([0, 1, 2, 2], [1, 2], [1, 2], {(0, 1): 2.5})
    results = search(turn, 0)
    assert results.path_cost_to(2) == 5.5
    assert results.path_turn_cost_to(2) == 2.5
    search(turn, 1, results=results)
    assert results.path_cost_to(2) == 2
    assert results.path_turn_cost_to(2) == 0
    # These layouts happen to have the same dimensions. No mode tag is needed.
    node = make_context(turn.fs, turn.heads, turn.costs)
    search(node, 0, results=results)
    assert results.root == 0
    assert results.path_cost_to(2) == 3
    assert results.path_turn_cost_to(2) == 0
    assert_state_tree(node, results)


def test_parallel_links_keep_distinct_turn_costs():
    context = make_context([0, 2, 3, 3], [1, 1, 2], [1, 2, 1], {(0, 2): 10})
    results = search(context, 0, 2)
    np.testing.assert_array_equal(results.path_links_to(2), [1, 2])
    assert results.path_cost_to(2) == 3
    assert results.terminal_states[1] == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize("reason", ["turn", "link", "overflow"])
def test_unreachable_and_partial_labels(reason):
    costs, penalty = [1, 1], np.inf
    if reason == "link":
        costs[1], penalty = np.inf, 0
    elif reason == "overflow":
        costs, penalty = [1e308, 1e308], 0
    context = make_context([0, 1, 2, 2], [1, 2], costs, {(0, 1): penalty})
    results = search(context, 0, 2)
    assert results.exhausted and not results.all_targets_reached
    assert results.path_links_to(2).size == 0
    assert np.isinf(results.path_cost_to(2))
    assert np.isinf(results.path_turn_cost_to(2))
    assert_state_tree(context, results)
    history = history_context()
    partial = search(history, 0, 1)
    assert not partial.reachable_to(2)
    assert not partial.reachable_to(3)
    assert_state_tree(history, partial)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("nodes", [1, 4])
def test_edgeless_and_intrazonal(turn, nodes):
    context = make_context(np.zeros(nodes + 1, dtype=np.uintp), [], [], turn=turn)
    results = search(context, 0)
    assert results.settled_count == 1
    assert results.exhausted
    assert_state_tree(context, results)
    search(context, nodes - 1, nodes - 1, results)
    assert results.path_links_to(nodes - 1).size == 0
    assert results.path_cost_to(nodes - 1) == results.path_turn_cost_to(nodes - 1) == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize("cost", [0, 1])
@pytest.mark.parametrize(
    "allow_uturns, override, reachable",
    [
        (True, None, True),
        (False, None, False),
        (False, 0, True),
        (False, 2, True),
        (True, np.inf, False),
    ],
)
def test_uturn_policy_and_zero_cost_cycles(cost, allow_uturns, override, reachable):
    turns = {(0, 2): np.inf}
    if override is not None:
        turns[1, 3] = override
    context = make_context([0, 1, 3, 4, 4], [1, 2, 3, 1], [cost] * 4, turns, allow_uturns=allow_uturns)
    results = search(context, 0, 3)
    assert results.reachable_to(3) == reachable
    if reachable:
        np.testing.assert_array_equal(results.path_links_to(3), [0, 1, 3, 2])
        assert results.path_cost_to(3) == 4 * cost + (override or 0)
        assert results.path_turn_cost_to(3) == (override or 0)
    assert_state_tree(context, results)


def test_turn_topology_copied_then_shared_by_independent_objectives():
    offsets = np.array([0, 99, 1, 99, 1])[::2]
    links = np.array([1, 99])[::2]
    penalties = np.array([2.5, 99])[::2]
    context = TurnBasedContext([0, 1, 2, 2], [1, 2], np.array([1.0, 2.0]), offsets, links, penalties)
    for source, output in (
        (offsets, context.turn_fs),
        (links, context.turn_to_links),
        (penalties, context.turn_penalties),
    ):
        assert not np.shares_memory(source, output)
        source[:] = 99
    other = context.with_costs(np.array([2.0, 3.0]))
    for name in ("fs", "heads", "tails", "turn_fs", "turn_to_links", "turn_penalties"):
        assert np.shares_memory(getattr(context, name), getattr(other, name))
    assert search(context, 0).path_cost_to(2) == 5.5
    del context
    assert search(other, 0).path_cost_to(2) == 7.5


@pytest.mark.parametrize(
    "offsets, links, penalties",
    [
        ([0, 0], [], []),
        ([1, 1, 1], [1], [0]),
        ([0, 2, 1], [1], [0]),
        ([0, 1, 1], [2], [0]),
        ([0, 1, 1], [-1], [0]),
        ([0, 1, 1], [1.0], [0]),
        ([0, 1, 1], [1], [-1]),
        ([0, 1, 1], [1], [np.nan]),
        ([0, 2, 2], [1, 1], [0, 0]),
        ([0, 1, 1], [0], [0]),
        ([0, 1, 1], None, [0]),
    ],
)
def test_invalid_turn_tables(offsets, links, penalties):
    with pytest.raises(ValueError):
        TurnBasedContext([0, 1, 2, 2], [1, 2], np.ones(2), offsets, links, penalties)


def expanded_oracle(context, turns, origin):
    """Build explicit transitions independently of the kernel's sparse-row merge."""
    graph = nx.DiGraph()
    root = context.link_count
    graph.add_nodes_from(range(root + 1))
    for first in range(context.fs[origin], context.fs[origin + 1]):
        if np.isfinite(context.costs[first]):
            graph.add_edge(root, first, weight=context.costs[first])
    for incoming in range(context.link_count):
        for outgoing in range(context.link_count):
            if context.heads[incoming] != context.tails[outgoing]:
                continue
            pair = incoming, outgoing
            if pair in turns:
                penalty = turns[pair]
            elif not context.allow_uturns and context.tails[incoming] == context.heads[outgoing]:
                continue
            else:
                penalty = 0
            weight = penalty + context.costs[outgoing]
            if np.isfinite(weight):
                graph.add_edge(incoming, outgoing, weight=weight)
    return nx.single_source_dijkstra_path_length(graph, root)


@pytest.mark.parametrize("seed", [17, 29, 42])
@pytest.mark.parametrize("allow_uturns", [False, True])
@pytest.mark.parametrize("use_hybrid", [False, True])
@pytest.mark.parametrize("turn_probability", [0.02, 0.35])
def test_random_multigraph_against_expanded_networkx(seed, allow_uturns, use_hybrid, turn_probability):
    rng = np.random.default_rng(seed)
    n = 8
    edges = [
        (a, b, float(rng.integers(0, 8))) for a in range(n) for b in range(n) for _ in range(2) if rng.random() < 0.12
    ]
    turns = {}
    for incoming, (_, via, _) in enumerate(edges):
        for outgoing, (tail, _, _) in enumerate(edges):
            if via == tail and rng.random() < turn_probability:
                turns[incoming, outgoing] = np.inf if rng.random() < 0.3 else float(rng.integers(0, 10))
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = make_context(
        fs, [b for _, b, _ in edges], [c for _, _, c in edges], turns,
        allow_uturns=allow_uturns, use_hybrid=use_hybrid,
    )
    results = allocate_results(context)
    for origin in range(n):
        oracle = expanded_oracle(context, turns, origin)
        for destination in range(n):
            expected = (
                0
                if origin == destination
                else min(
                    (
                        oracle.get(link, np.inf)
                        for link in range(context.link_count)
                        if context.heads[link] == destination
                    ),
                    default=np.inf,
                )
            )
            search(context, origin, destination, results)
            assert results.path_cost_to(destination) == expected
            assert_state_tree(context, results)
            for state in results.settlement_order[: results.settled_count]:
                incoming = results.root if state == results.root else results.connectors[state]
                assert results.distances[state] == oracle[incoming]
            if not results.reachable_to(destination):
                continue
            links = results.path_links_to(destination)
            penalty = 0
            for incoming, outgoing in zip(links[:-1], links[1:], strict=True):
                assert context.heads[incoming] == context.tails[outgoing]
                if not allow_uturns and context.tails[incoming] == context.heads[outgoing]:
                    assert (incoming, outgoing) in turns
                penalty += turns.get((incoming, outgoing), 0)
            assert context.costs[links].sum() + penalty == expected
            assert results.path_turn_cost_to(destination) == penalty


def test_hybrid_reduces_states_away_from_turn_controls():
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8)).to_directed()
    nodes = len(graph)
    edges = sorted(graph.edges)
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _ in edges], minlength=nodes))]
    heads = [b for _, b in edges]
    costs = np.ones(len(edges))
    turns = {(0, int(fs[heads[0]])): 2.0}
    full = make_context(fs, heads, costs, turns, allow_uturns=False)
    hybrid = make_context(fs, heads, costs, turns, allow_uturns=False, use_hybrid=True)
    reference, result = search(full, 0), search(hybrid, 0)

    assert result.settled_count < reference.settled_count / 2
    assert_state_tree(hybrid, result)
    for destination in range(nodes):
        assert result.path_cost_to(destination) == reference.path_cost_to(destination)


def test_hybrid_retains_winning_link_and_reuses_labels_with_new_costs():
    context = make_context([0, 2, 3, 4, 4], [1, 2, 3, 3], [10, 1, 1, 1], turn=True, use_hybrid=True)
    results = search(context, 0, 3)
    assert results.terminal_states[3] == 2
    assert results.connectors[2] == 3
    np.testing.assert_array_equal(results.path_links_to(3), [1, 3])
    assert results.path_cost_to(3) == 2
    assert_state_tree(context, results)

    rebound = context.with_costs(np.array([1.0, 10.0, 1.0, 1.0]))
    search(rebound, 0, 3, results)
    np.testing.assert_array_equal(results.path_links_to(3), [0, 2])
    assert results.path_cost_to(3) == 2
    assert_state_tree(rebound, results)
