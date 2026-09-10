"""Cascade loading of the common state tree, independent of production AoN."""

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_workspace import AoNWorkspace
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext


def path_loads(context, results, demand):
    expected = np.zeros((context.link_count, demand.shape[1]))
    for destination, row in enumerate(demand):
        for link in results.path_links_to(destination):
            expected[link] += row
    return expected


@pytest.mark.parametrize("penalty", [1., 10., np.inf])
def test_turn_history(penalty):
    # Cheapest arrival at 1 is not necessarily the arrival used en route to 3.
    context = TurnBasedContext(
        [0, 2, 3, 4, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1],
        [0, 1, 1, 1, 1], [2], [penalty],
    )
    results = dijkstra(context, 0, None)
    demand = np.array([[np.nan, np.inf], [2, 20], [3, 30], [5, 50], [np.nan, np.inf]])
    demand.flags.writeable = False
    loads = np.zeros((4, 2))
    assert results.network_loading(demand, loads) is loads
    expected = [7, 3, 5, 0] if penalty == 1 else [2, 8, 5, 5]
    np.testing.assert_array_equal(loads, np.array(expected)[:, None] * [1, 10])
    np.testing.assert_array_equal(loads, path_loads(context, results, demand))
    np.testing.assert_array_equal(results.workspace.state_loads[results.root], [10, 100])


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
@pytest.mark.parametrize("classes", [0, 1, 3, 37])
def test_all_origins_accumulate_and_reset(context_type, classes):
    # Parallel links, self loop, zero-cost cycle, and isolated node.
    context = context_type([0, 4, 6, 7, 7, 7], [0, 1, 1, 2, 0, 2, 3], [0, 2, 0, 10, 0, 1, 1])
    results = context.make_results()
    assert isinstance(results.workspace, AoNWorkspace)
    assert results.workspace.state_loads is None
    demand = np.arange(5 * 5 * classes, dtype=np.float64).reshape(5, 5, classes)
    snapshot = demand.copy()
    results.workspace.prepare_loading(classes)
    scratch = results.workspace.state_loads
    assert scratch.shape == (context.state_count, classes)
    assert not scratch.flags.writeable
    loads = np.zeros((context.link_count, classes))
    for _ in range(2):
        loads.fill(0)
        expected = np.zeros_like(loads)
        for origin in range(5):
            dijkstra(context, origin, None, results)
            results.network_loading(demand[origin], loads)
            expected += path_loads(context, results, demand[origin])
            np.testing.assert_array_equal(loads, expected)
            assert results.workspace.state_loads.ctypes.data == scratch.ctypes.data
        np.testing.assert_array_equal(scratch, 0)  # Last origin is isolated.
    np.testing.assert_array_equal(demand, snapshot)


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_partial_search_empty_demand_and_scratch_lifetimes(context_type):
    context = context_type([0, 1, 2, 2], [1, 2], [1, 1])
    results = context.make_results()
    loads = np.full((2, 2), 7.)
    demand = np.array([[100., 100.], [2, 3], [5, 8]])
    results.network_loading(demand, loads)  # Before a search: no paths.
    np.testing.assert_array_equal(loads, 7.)
    dijkstra(context, 0, None, results)
    results.network_loading(demand, loads)
    np.testing.assert_array_equal(loads, [[14, 18], [12, 15]])
    scratch = results.workspace.state_loads
    dijkstra(context, 0, 1, results)
    loads.fill(0)
    results.network_loading(demand, loads)
    np.testing.assert_array_equal(loads, [[2, 3], [0, 0]])
    assert results.workspace.state_loads.ctypes.data == scratch.ctypes.data
    unsettled = results.predecessors == results.sentinel
    unsettled[results.root] = False
    np.testing.assert_array_equal(scratch[unsettled], 0)
    results.network_loading(np.empty((0, 2)), loads)
    np.testing.assert_array_equal(loads, [[2, 3], [0, 0]])
    np.testing.assert_array_equal(scratch, 0)
    dijkstra(context, 2, 2, results)
    results.network_loading(demand, loads)
    np.testing.assert_array_equal(loads, [[2, 3], [0, 0]])

    # Skims and loading have independent scratch allocations.
    results.skim_fields([np.ones(2)])
    skim_snapshot = results.workspace.state_skims.copy()
    dijkstra(context, 0, None, results)
    results.network_loading(demand, loads)
    np.testing.assert_array_equal(results.workspace.state_skims, skim_snapshot)
    snapshot = scratch.copy()
    results.workspace.prepare_loading(3)
    assert results.workspace.loading_class_count == 3
    np.testing.assert_array_equal(scratch, snapshot)
    new_view = results.workspace.state_loads
    del results, context
    gc.collect()
    np.testing.assert_array_equal(scratch, snapshot)
    np.testing.assert_array_equal(new_view, 0)


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_centroids_use_intermediate_states(context_type):
    # Only nodes 0 and 1 have demand rows. Path 0->2->3->1 uses other states.
    context = context_type([0, 1, 1, 2, 3], [2, 3, 1], [1, 1, 1])
    results = dijkstra(context, 0, range(2))
    loads = np.zeros((3, 2))
    results.network_loading(np.array([[999., 999.], [4, 7]]), loads)
    np.testing.assert_array_equal(loads, [[4, 7]] * 3)


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_edgeless_graph(context_type):
    context = context_type([0, 0, 0], [], [])
    results = dijkstra(context, 1, None)
    loads = np.empty((0, 2))
    assert results.network_loading(np.ones((2, 2)), loads) is loads
    np.testing.assert_array_equal(results.workspace.state_loads, 0)


def test_ieee_demand_only_propagates_on_its_path():
    context = NodeBasedContext([0, 2, 3, 3, 3], [1, 2, 3], [1, 1, 1])
    results = dijkstra(context, 0, None)
    demand = np.array([[0., 0., 0.], [2, 3, -1], [4, 5, -2], [np.nan, np.inf, -3]])
    loads = np.zeros((3, 3))
    results.network_loading(demand, loads)
    np.testing.assert_array_equal(loads, [[np.nan, np.inf, -4], [4, 5, -2], [np.nan, np.inf, -3]])


def test_validation():
    context = NodeBasedContext([0, 1, 2], [1, 0], [1, 1])
    results = dijkstra(context, 0, None)
    demand = np.ones((2, 2))
    loads = np.zeros((2, 2))
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    for bad in (
        None, [[1., 1.]] * 2, demand.astype(np.float32), demand.astype(np.int64),
        np.ones(2), np.ones((3, 2)), np.ones((2, 2, 1)), np.ones((4, 2))[::2],
        np.ones((2, 4))[:, ::2], np.asfortranarray(demand), unaligned,
    ):
        with pytest.raises((TypeError, ValueError)):
            results.network_loading(bad, loads)
    readonly = loads.copy()
    readonly.flags.writeable = False
    for bad in (
        None, [[0., 0.]] * 2, loads.astype(np.float32), np.zeros((1, 2)),
        np.zeros((2, 1)), np.zeros(4), np.zeros((4, 2))[::2],
        np.asfortranarray(loads), readonly, unaligned,
    ):
        with pytest.raises((TypeError, ValueError)):
            results.network_loading(demand, bad)
    with pytest.raises(ValueError, match="overlap"):
        results.network_loading(demand, demand)
    results.workspace.prepare_loading(2)
    with pytest.raises(ValueError, match="overlap"):
        results.network_loading(results.workspace.state_loads, loads)
    with pytest.raises(ValueError, match="nonnegative"):
        results.workspace.prepare_loading(-1)
    with pytest.raises(TypeError):
        results.workspace.prepare_loading(1.5)
    # Validation never touched the accumulator.
    np.testing.assert_array_equal(loads, 0)


def random_context(turns, seed):
    rng = np.random.default_rng(seed)
    n, m = 9, 40
    tails = np.sort(rng.integers(0, n - 1, m))
    heads = rng.integers(0, n - 1, m)
    fs = np.r_[0, np.cumsum(np.bincount(tails, minlength=n))]
    costs = rng.integers(0, 6, m).astype(np.float64)
    if not turns:
        return NodeBasedContext(fs, heads, costs)
    turn_fs, to, penalties = [0], [], []
    for incoming in range(m):
        for outgoing in range(fs[heads[incoming]], fs[heads[incoming] + 1]):
            if rng.random() < 0.5:
                to.append(outgoing)
                penalties.append(rng.choice([0., 1., 4., np.inf]))
        turn_fs.append(len(to))
    return TurnBasedContext(fs, heads, costs, turn_fs, to, penalties, allow_uturns=False)


@pytest.mark.parametrize("turns", [False, True])
@pytest.mark.parametrize("seed", [17, 71, 123])
def test_random_multigraphs_against_path_loading(turns, seed):
    context = random_context(turns, seed)
    rng = np.random.default_rng(seed)
    results = context.make_results()
    for count in (0, 3, context.node_count):
        loads = np.zeros((context.link_count, 5))
        expected = loads.copy()
        for origin in range(context.node_count):
            for targets in (None, [origin], [0, 7]):
                demand = rng.random((count, 5))
                dijkstra(context, origin, targets, results)
                results.network_loading(demand, loads)
                expected += path_loads(context, results, demand)
                np.testing.assert_allclose(loads, expected)


@pytest.mark.parametrize("turns", [False, True])
def test_thread_slices_accumulate_and_reduce(turns):
    context = random_context(turns, 42)
    threads, classes = 3, 4
    demand = np.random.default_rng(42).random((context.node_count, context.node_count, classes))
    demand.flags.writeable = False
    thread_loads = np.zeros((threads, context.link_count, classes))
    workers = [context.make_results() for _ in range(threads)]
    for results in workers:
        results.workspace.prepare_loading(classes)

    def load_origins(tid):
        results = workers[tid]
        for origin in range(tid, context.node_count, threads):
            dijkstra(context, origin, None, results)
            results.network_loading(demand[origin], thread_loads[tid])

    expected = np.zeros((context.link_count, classes))
    for origin in range(context.node_count):
        results = dijkstra(context, origin, None)
        expected += path_loads(context, results, demand[origin])
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(2):
            thread_loads.fill(0)
            list(pool.map(load_origins, range(threads)))
            np.testing.assert_allclose(thread_loads.sum(axis=0), expected)
    assert len({r.workspace.state_loads.ctypes.data for r in workers}) == threads
