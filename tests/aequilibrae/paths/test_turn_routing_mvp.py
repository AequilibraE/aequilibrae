"""Turn histories checked against an independently expanded state graph."""

import networkx as nx
import numpy as np
import pytest

from aequilibrae.paths.cython.context import TurnBasedContext
from .routing_helpers import allocate_results, assert_state_tree, history_context, make_context, search


@pytest.mark.parametrize("penalty", [10, np.inf])
def test_nonterminal_arrival_history(penalty):
    context = history_context(penalty)
    results = search(context, 0, [1, 3])
    assert results.root == context.link_count
    assert results.terminal_states[1] == 0
    assert results.predecessors[results.terminal_states[3]] == 3
    np.testing.assert_array_equal(results.path_links_to(1), [0])
    np.testing.assert_array_equal(results.path_links_to(3), [1, 3, 2])
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
def test_random_multigraph_against_expanded_networkx(seed, allow_uturns):
    rng = np.random.default_rng(seed)
    n = 8
    edges = [
        (a, b, float(rng.integers(0, 8))) for a in range(n) for b in range(n) for _ in range(2) if rng.random() < 0.12
    ]
    turns = {}
    for incoming, (_, via, _) in enumerate(edges):
        for outgoing, (tail, _, _) in enumerate(edges):
            if via == tail and rng.random() < 0.35:
                turns[incoming, outgoing] = np.inf if rng.random() < 0.3 else float(rng.integers(0, 10))
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = make_context(fs, [b for _, b, _ in edges], [c for _, _, c in edges], turns, allow_uturns=allow_uturns)
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
                assert results.distances[state] == oracle[state]
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
