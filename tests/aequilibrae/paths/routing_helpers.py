"""Test setup and path walks, kept outside the internal routing interface."""

import numpy as np

from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults


def make_context(fs, heads, costs, turns=None, *, turn=False, **options):
    costs = np.asarray(costs, dtype=np.float64)
    if not turn and turns is None:
        return NodeBasedContext(fs, heads, costs, **options)
    turns = {} if turns is None else turns
    pairs = sorted(turns)
    offsets = np.r_[0, np.cumsum(np.bincount([a for a, _ in pairs], minlength=len(heads)))]
    return TurnBasedContext(fs, heads, costs, offsets, [b for _, b in pairs],
                            [turns[pair] for pair in pairs], **options)


def history_context(penalty=10.0, *, turn=True):
    # Cheapest arrival at 1 is not the arrival used on the best route to 3.
    turns = {(0, 2): penalty} if turn else None
    return make_context([0, 2, 3, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1], turns, turn=turn)


def allocate_results(context):
    return SearchResults(context.node_count, context.state_count, context.link_count)


def search(context, origin, targets=None, results=None):
    mask = None
    if targets is not None:
        mask = np.zeros(context.node_count, dtype=np.bool_)
        mask[np.atleast_1d(targets)] = True
    query = SearchQuery(context.node_count, origin, mask)
    if results is None:
        results = allocate_results(context)
    return dijkstra(context, query, results)


def assert_state_tree(context, results):
    order = results.settlement_order[:results.settled_count]
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
    assert np.all(results.settlement_order[results.settled_count:] == results.sentinel)
    for node, terminal in enumerate(results.terminal_states):
        if terminal != results.sentinel:
            assert int(terminal) in rank
            if terminal == results.root:
                assert node == results.origin
            else:
                assert context.heads[results.connectors[terminal]] == node


def path_walk_outputs(context, demand, fields=(), penalty_fields=(), selected_links=()):
    """Compare downstream kernels against separate OD-by-OD link walks."""
    zones, _, classes = demand.shape
    loads = np.zeros((context.link_count, classes))
    skims = np.full((zones, len(fields), zones), np.inf)
    selected_loads = np.zeros((len(selected_links), context.link_count, classes))
    selected_od = np.zeros((len(selected_links), zones, zones, classes))
    total = 0.0
    results = allocate_results(context)
    for origin in range(zones):
        search(context, origin, results=results)
        for destination in range(zones):
            if not results.reachable_to(destination):
                continue
            links = results.path_links_to(destination)
            penalty = results.path_turn_cost_to(destination)
            for field_index, field in enumerate(fields):
                skims[origin, field_index, destination] = field[links].sum()
                if penalty_fields and penalty_fields[field_index]:
                    skims[origin, field_index, destination] += penalty
            if origin == destination:
                continue
            row = demand[origin, destination]
            total += row.sum() * penalty
            # A turn path may use a link more than once, so use scalar additions.
            for link in links:
                loads[link] += row
            for selection, members in enumerate(selected_links):
                if any(link in members for link in links):
                    selected_od[selection, origin, destination] = row
                    for link in links:
                        selected_loads[selection, link] += row
    return loads, skims, total, selected_loads, selected_od
