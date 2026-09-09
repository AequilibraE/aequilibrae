"""Turn-context MVP tests, independent of production Graph and assignment."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest

from aequilibrae.paths.cython.graph_context import NodeBasedContext
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.graph_context import TurnBasedContext


def make_context(fs, heads, costs, turns=None, *, allow_uturns=True):
    turns = turns or {}
    entries = sorted(turns)
    turn_fs = np.r_[0, np.cumsum(np.bincount([a for a, _ in entries], minlength=len(heads)))]
    return TurnBasedContext(
        fs,
        heads,
        costs,
        turn_fs,
        [b for _, b in entries],
        [turns[pair] for pair in entries],
        allow_uturns=allow_uturns,
    )


def history_context(penalty=10):
    # Links: 0: 0->1, 1: 0->2, 2: 1->3, 3: 2->1.
    # Best path to node 1 uses link 0, but best path to node 3 must not!
    return make_context([0, 2, 3, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1], {(0, 2): penalty})


def assert_state_tree(context, results):
    order = results.reached_first[: results.settled_count]
    rank = {int(state): i for i, state in enumerate(order)}
    assert len(rank) == results.settled_count
    assert order[0] == results.root
    assert results.predecessors[results.root] == results.sentinel
    assert results.connectors[results.root] == results.sentinel
    assert results.distances[results.root] == results.turn_costs[results.root] == 0
    for state in order[1:]:
        parent = results.predecessors[state]
        assert rank[int(parent)] < rank[int(state)]
        assert results.connectors[state] < context.link_count
        assert results.distances[parent] <= results.distances[state]
    unfinalized = np.ones(context.state_count, dtype=bool)
    unfinalized[order] = False
    assert np.all(results.predecessors[unfinalized] == results.sentinel)
    assert np.all(results.connectors[unfinalized] == results.sentinel)
    assert np.all(np.isinf(results.distances[unfinalized]))
    assert np.all(np.isinf(results.turn_costs[unfinalized]))
    assert np.all(results.reached_first[results.settled_count :] == results.sentinel)
    for node, terminal in enumerate(results.terminal_states):
        if terminal != results.sentinel:
            assert int(terminal) in rank
            if terminal == results.root:
                assert node == results.origin
            else:
                assert context.heads[results.connectors[terminal]] == node


def test_initialization():
    context = history_context()
    results = context.make_results()
    assert isinstance(results, SearchResults)
    assert results.context is context
    assert results.state_count == context.state_count == context.link_count + 1
    assert results.terminal_states.shape == (context.node_count,)
    assert results.root is results.origin is results.destination is None
    assert results.settled_count == 0
    assert not results.reachable
    assert np.isinf(results.path_cost)
    assert np.isinf(results.path_turn_cost)
    assert np.all(results.terminal_states == results.sentinel)
    for name in ("predecessors", "connectors", "reached_first"):
        array = getattr(results, name)
        assert array.shape == (context.state_count,)
        assert np.all(array == results.sentinel)
    for name in ("distances", "turn_costs"):
        array = getattr(results, name)
        assert array.shape == (context.state_count,)
        assert array.dtype == np.float64
        assert np.all(np.isinf(array))


@pytest.mark.parametrize("penalty", [10, np.inf])
def test_different_arrival_histories_are_preserved(penalty):
    context = history_context(penalty)
    results = dijkstra(context, 0, 3)
    np.testing.assert_array_equal(results.path_nodes, [0, 2, 1, 3])
    np.testing.assert_array_equal(results.path_links, [1, 3, 2])
    assert results.path_cost == 3
    assert results.path_turn_cost == 0
    assert results.root == context.link_count
    assert results.terminal_states[1] == 0
    assert results.terminal_states[3] == 2
    assert results.predecessors[2] == 3  # Not terminal_states[1].
    assert results.settled_count > context.node_count
    assert_state_tree(context, results)


def test_unavoidable_penalty_and_first_link_has_no_turn_cost():
    context = make_context([0, 1, 2, 2], [1, 2], [1, 2], {(0, 1): 2.5})
    results = dijkstra(context, 0, 2)
    assert results.path_cost == 5.5
    assert results.path_turn_cost == 2.5
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2])
    np.testing.assert_array_equal(results.turn_costs, [0, 2.5, 0])
    assert_state_tree(context, results)
    dijkstra(context, 1, 2, results)
    assert results.path_cost == 2
    assert results.path_turn_cost == 0
    np.testing.assert_array_equal(results.path_links, [1])
    assert_state_tree(context, results)


def test_parallel_links_keep_distinct_turn_costs():
    context = make_context([0, 2, 3, 3], [1, 1, 2], [1, 2, 1], {(0, 2): 10})
    results = dijkstra(context, 0, 2)
    np.testing.assert_array_equal(results.path_links, [1, 2])
    assert results.path_cost == 3
    assert results.terminal_states[1] == 0
    assert results.predecessors[2] == 1
    assert_state_tree(context, results)


def test_multiple_destinations_count_physical_nodes_not_arrival_states():
    context = history_context()
    results = dijkstra(context, 0, [1, 3])
    assert results.reachable
    assert results.destination_count == results.reached_destination_count == 2
    assert results.terminal_states[1] == 0
    assert results.terminal_states[3] == 2
    np.testing.assert_array_equal(results.path_nodes_to(1), [0, 1])
    np.testing.assert_array_equal(results.path_nodes_to(3), [0, 2, 1, 3])
    assert results.path_cost_to(1) == 1
    assert results.path_cost_to(3) == 3
    assert_state_tree(context, results)


def test_empty_destination_mask_disables_early_exit():
    context = history_context()
    results = dijkstra(context, 0, np.zeros(context.node_count, dtype=bool))
    assert results.destination_count == results.reached_destination_count == 0
    assert results.all_destinations_reached
    assert results.settled_count == context.state_count
    assert_state_tree(context, results)


def test_early_exit_clears_tentative_state_labels():
    context = history_context()
    results = dijkstra(context, 0, 1)
    assert results.settled_count == 2
    assert results.terminal_states[2] == results.sentinel
    assert results.terminal_states[3] == results.sentinel
    assert results.predecessors[1] == results.sentinel
    assert np.isinf(results.distances[1])
    assert np.isinf(results.turn_costs[1])
    assert_state_tree(context, results)


@pytest.mark.parametrize("reason", ["turn", "link", "overflow"])
def test_unreachable(reason):
    costs = [1, 1]
    penalty = np.inf
    if reason == "link":
        costs[1], penalty = np.inf, 0
    elif reason == "overflow":
        costs, penalty = [1e308, 1e308], 0
    context = make_context([0, 1, 2, 2], [1, 2], costs, {(0, 1): penalty})
    results = dijkstra(context, 0, 2)
    assert not results.reachable
    assert results.path_nodes.size == results.path_links.size == 0
    assert np.isinf(results.path_cost)
    assert np.isinf(results.path_turn_cost)
    assert_state_tree(context, results)


@pytest.mark.parametrize("nodes", [1, 4])
def test_edgeless_and_intrazonal(nodes):
    context = TurnBasedContext(np.zeros(nodes + 1, dtype=np.uintp), [], [])
    results = dijkstra(context, 0, nodes - 1)
    assert results.root == 0
    assert results.settled_count == 1
    assert results.reachable == (nodes == 1)
    assert_state_tree(context, results)
    dijkstra(context, nodes - 1, nodes - 1, results)
    np.testing.assert_array_equal(results.path_nodes, [nodes - 1])
    assert results.path_links.size == 0
    assert results.path_cost == results.path_turn_cost == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize(
    "allow_uturns, override, reachable",
    [(True, None, True), (False, None, False), (False, 0, True), (False, 2, True), (True, np.inf, False)],
)
def test_uturn_policy_and_explicit_override(allow_uturns, override, reachable):
    # Must pass through node 1 twice to avoid a prohibited 0 -> 1 -> 3 turn.
    turns = {(0, 2): np.inf}
    if override is not None:
        turns[1, 3] = override
    context = make_context([0, 1, 3, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1], turns, allow_uturns=allow_uturns)
    results = dijkstra(context, 0, 3)
    assert results.reachable == reachable
    if reachable:
        np.testing.assert_array_equal(results.path_nodes, [0, 1, 2, 1, 3])
        np.testing.assert_array_equal(results.path_links, [0, 1, 3, 2])
        assert results.path_cost == 4 + (override or 0)
        assert results.path_turn_cost == (override or 0)
    assert_state_tree(context, results)


def test_zero_cost_cycle():
    context = make_context([0, 1, 3, 4, 4], [1, 2, 3, 1], [0, 0, 0, 0], {(0, 2): np.inf})
    results = dijkstra(context, 0, 3)
    assert results.reachable
    assert results.path_cost == 0
    np.testing.assert_array_equal(results.path_nodes, [0, 1, 2, 1, 3])
    assert_state_tree(context, results)


def test_reuse_keeps_views_and_resets_all_arrays():
    context = history_context()
    results = dijkstra(context, 0, 3)
    names = ("predecessors", "connectors", "reached_first", "distances", "turn_costs", "terminal_states")
    arrays = [getattr(results, name) for name in names]
    snapshots = [array.copy() for array in arrays]
    for origin, destination in [(3, 0), (0, 1), (2, 2), (0, 3)]:
        assert dijkstra(context, origin, destination, results) is results
        assert_state_tree(context, results)
        for name, array in zip(names, arrays, strict=True):
            assert np.shares_memory(array, getattr(results, name))
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_views_own_lifetimes_and_reject_writes():
    context = history_context()
    results = dijkstra(context, 0, 3)
    views = [
        getattr(context, name)
        for name in ("fs", "heads", "costs", "link_ids", "tails", "turn_fs", "turn_to_links", "turn_penalties")
    ]
    views += [
        getattr(results, name)
        for name in ("predecessors", "connectors", "reached_first", "distances", "turn_costs", "terminal_states")
    ]
    for view in views:
        assert not view.flags.owndata
        with pytest.raises(ValueError):
            view[0] = 0
        with pytest.raises(ValueError):
            view.flags.writeable = True
    refs = [weakref.ref(view.base) for view in views]
    snapshots = [view.copy() for view in views]
    del context
    gc.collect()
    dijkstra(results.context, 0, 3, results)
    del results
    gc.collect()
    for view, snapshot in zip(views, snapshots, strict=True):
        np.testing.assert_array_equal(view, snapshot)
    assert all(ref() is not None for ref in refs)
    del view, views
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_turn_inputs_are_copied_from_strided_arrays():
    turn_fs = np.array([0, 99, 1, 99, 1, 99])[::2]
    to_links = np.array([1, 99])[::2]
    penalties = np.array([2.5, 99])[::2]
    context = TurnBasedContext([0, 1, 2, 2], [1, 2], [1, 2], turn_fs, to_links, penalties)
    for source, output in (
        (turn_fs, context.turn_fs),
        (to_links, context.turn_to_links),
        (penalties, context.turn_penalties),
    ):
        assert output.flags.c_contiguous
        assert not np.shares_memory(source, output)
        source[:] = 99
    assert dijkstra(context, 0, 2).path_cost == 5.5


@pytest.mark.parametrize(
    "turn_fs, to_links, penalties",
    [
        ([0, 0], [], []),
        ([1, 1, 1], [1], [0]),
        ([0, 2, 1], [1], [0]),
        ([0, 0, 0], [1], [0]),
        ([0, 1, 1], [2], [0]),
        ([0, 1, 1], [-1], [0]),
        ([0, 1, 1], [1.0], [0]),
        ([0.0, 1.0, 1.0], [1], [0]),
        ([[0, 1, 1]], [1], [0]),
        ([0, 1, 1], [[1]], [0]),
        ([0, 1, 1], [1], []),
        ([0, 1, 1], [1], [[0]]),
        ([0, 1, 1], [1], [-1]),
        ([0, 1, 1], [1], [-np.inf]),
        ([0, 1, 1], [1], [np.nan]),
        ([0, 2, 2], [1, 1], [0, 0]),  # Duplicate outgoing link.
        ([0, 1, 1], [0], [0]),  # Links do not share a via node.
        ([0, 1, 1], None, [0]),
    ],
)
def test_invalid_turn_data(turn_fs, to_links, penalties):
    with pytest.raises(ValueError):
        TurnBasedContext([0, 1, 2, 2], [1, 2], [1, 1], turn_fs, to_links, penalties)


def test_unsorted_turn_row():
    with pytest.raises(ValueError, match="strictly increasing"):
        TurnBasedContext([0, 1, 3, 3], [1, 2, 2], [1, 1, 1], [0, 2, 2, 2], [2, 1], [0, 0])


def test_context_compatibility_and_dispatch():
    turn = history_context()
    node = NodeBasedContext(turn.fs, turn.heads, turn.costs)
    for a, b in [(turn, node), (node, turn), (turn, history_context())]:
        with pytest.raises(ValueError, match="different context"):
            dijkstra(a, 0, 3, b.make_results())
    for invalid in (None, object()):
        with pytest.raises(TypeError):
            dijkstra(invalid, 0, 1)
        with pytest.raises(TypeError):
            SearchResults(invalid)
    for origin, destination in [(-1, 0), (0, 4), (2**100, 0)]:
        with pytest.raises(ValueError, match="node range"):
            dijkstra(turn, origin, destination)
    for context in (turn, node):
        result = dijkstra(context, np.int64(0), np.uint64(3))
        assert_state_tree(context, result)
    assert dijkstra(node, 0, 3).path_cost == 2
    assert dijkstra(node, 0, 3).path_turn_cost == 0


def test_shared_context_separate_worker_results():
    n = 2000
    context = TurnBasedContext(np.r_[np.arange(n), n - 1], np.arange(1, n), np.ones(n - 1))
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


def expanded_oracle(context, turns, origin):
    """Independent explicit state graph, materialized only in tests."""
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
def test_random_multigraph_against_expanded_networkx(seed, allow_uturns):
    rng = np.random.default_rng(seed)
    n = 8
    edges = []
    for a in range(n):
        for b in range(n):
            if rng.random() < 0.22:
                for _ in range(rng.integers(1, 3)):
                    edges.append((a, b, float(rng.integers(0, 8))))
    turns = {}
    for incoming, (_, via, _) in enumerate(edges):
        for outgoing, (tail, _, _) in enumerate(edges):
            if via == tail and rng.random() < 0.35:
                turns[incoming, outgoing] = np.inf if rng.random() < 0.3 else float(rng.integers(0, 10))
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = make_context(
        fs, [b for _, b, _ in edges], [cost for _, _, cost in edges], turns, allow_uturns=allow_uturns
    )
    results = context.make_results()
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
            dijkstra(context, origin, destination, results)
            assert results.reachable == np.isfinite(expected)
            assert results.path_cost == expected
            assert_state_tree(context, results)
            for state in results.reached_first[: results.settled_count]:
                assert results.distances[state] == oracle[state]
            if not results.reachable:
                continue
            links = results.path_links
            actual_penalty = 0
            for incoming, outgoing in zip(links[:-1], links[1:], strict=True):
                assert context.heads[incoming] == context.tails[outgoing]
                if not allow_uturns and context.tails[incoming] == context.heads[outgoing]:
                    assert (incoming, outgoing) in turns
                actual_penalty += turns.get((incoming, outgoing), 0)
            assert context.costs[links].sum() + actual_penalty == expected
            assert results.path_turn_cost == actual_penalty
            assert results.path_nodes[0] == origin
            assert results.path_nodes[-1] == destination


def test_turn_context_without_penalties_matches_node_context():
    turn = TurnBasedContext([0, 2, 3, 4, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1])
    node = NodeBasedContext(turn.fs, turn.heads, turn.costs)
    for origin in range(node.node_count):
        for destination in range(node.node_count):
            a = dijkstra(node, origin, destination)
            b = dijkstra(turn, origin, destination)
            assert a.reachable == b.reachable
            assert a.path_cost == b.path_cost
            if a.reachable:
                assert a.path_turn_cost == b.path_turn_cost == 0
