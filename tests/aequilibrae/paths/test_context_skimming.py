"""Single-origin state-tree skims, deliberately independent of Graph/AoN."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.aon_workspace import AoNWorkspace
from aequilibrae.paths.cython.skimming_context import SkimmingContext


def history_context(penalty=10):
    # 0->1 is cheapest at node 1, but 0->2->1->3 is cheapest at node 3.
    # A penalty of 1 instead selects 0->1->3, paying a positive turn cost.
    # Node 4 is unreachable.
    return TurnBasedContext(
        [0, 2, 3, 4, 4, 4], [1, 2, 3, 1], [1, 1, 1, 1],
        [0, 1, 1, 1, 1], [2], [penalty],
    )


def assert_skims_match_paths(context, results, fields, skims):
    expected = np.full((context.node_count, len(fields)), np.inf)
    for node in range(context.node_count):
        if results.reachable_to(node):
            links = results.path_links_to(node)
            expected[node] = [field[links].sum() for field in fields]
    np.testing.assert_allclose(skims, expected)
    np.testing.assert_array_equal(
        results.skim_costs()[:, 0], [results.path_cost_to(n) for n in range(context.node_count)]
    )
    np.testing.assert_array_equal(
        results.skim_turn_costs()[:, 0], [results.path_turn_cost_to(n) for n in range(context.node_count)]
    )
    for state in range(context.state_count):
        row = results.workspace.state_skims[state]
        if state == results.root:
            np.testing.assert_array_equal(row, np.zeros(len(fields)))
        elif results.predecessors[state] == results.sentinel:
            assert np.isinf(row).all()
        else:
            parent = results.predecessors[state]
            link = results.connectors[state]
            np.testing.assert_allclose(row, results.workspace.state_skims[parent] + [f[link] for f in fields])


@pytest.mark.parametrize("penalty", [10, np.inf])
def test_turn_history_uses_states_not_node_predecessors(penalty):
    context = history_context(penalty)
    results = dijkstra(context, 0, None)
    fields = [np.array([10., 20., 30., 40.]), np.array([-1., 3., 4., 5.])]
    skims = results.skim_fields(fields)
    np.testing.assert_array_equal(skims, [[0, 0], [10, -1], [20, 3], [90, 12], [np.inf, np.inf]])
    # Both arrivals at node 1 are retained in state scratch, but only one is
    # projected to that physical node. The other is used en route to node 3.
    assert results.workspace.state_skims[0, 0] == 10
    assert results.workspace.state_skims[3, 0] == 60
    assert_skims_match_paths(context, results, fields, skims)


def test_explicit_cost_and_penalty_skims():
    context = history_context(1)
    results = dijkstra(context, 0, None)
    fields = [context.costs, np.array([100., 200., 300., 400.])]
    skims = results.skim_fields(fields)
    assert skims[3, 0] == 2  # Even the costs pointer is an ordinary link field here.
    assert skims[3, 1] == 400  # No turn penalty added to arbitrary fields.
    np.testing.assert_array_equal(results.skim_costs()[:, 0], [0, 1, 1, 3, np.inf])
    np.testing.assert_array_equal(results.skim_turn_costs()[:, 0], [0, 0, 0, 1, np.inf])
    assert_skims_match_paths(context, results, fields, skims)


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
@pytest.mark.parametrize("field_count", [0, 1, 3, 37])
def test_all_od_pairs_and_arbitrary_field_count(context_type, field_count):
    # Parallel links, a self loop, zero-cost cycles, and an isolated node.
    context = context_type([0, 4, 6, 7, 7, 7], [0, 1, 1, 2, 0, 2, 3], [0, 2, 0, 10, 0, 1, 1])
    fields = [np.arange(context.link_count, dtype=np.float64) + k for k in range(field_count)]
    cube = np.full((context.node_count, context.node_count, field_count), -123., dtype=np.float64)
    costs = np.full((context.node_count, context.node_count), -123., dtype=np.float64)
    turns = costs.copy()
    results = context.make_results()
    for origin in range(context.node_count):
        dijkstra(context, origin, None, results)
        out = cube[origin]
        assert results.skim_fields(fields, out=out) is out
        cost_out = costs[origin].reshape(-1, 1)
        assert results.skim_costs(out=cost_out) is cost_out
        turn_out = turns[origin].reshape(-1, 1)
        assert results.skim_turn_costs(out=turn_out) is turn_out
        assert_skims_match_paths(context, results, fields, out)
        np.testing.assert_array_equal(costs[origin], results.skim_costs()[:, 0])
        np.testing.assert_array_equal(turns[origin], results.skim_turn_costs()[:, 0])
    assert not np.any(cube == -123.)
    assert np.all(costs.diagonal() == 0)
    assert np.all(turns.diagonal() == 0)
    assert np.isinf(costs[4, :4]).all()


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_before_search_partial_search_and_workspace_reuse(context_type):
    context = context_type([0, 1, 2, 2], [1, 2], [1, 1])
    fields = [np.array([2., 3.])]
    results = context.make_results()
    assert np.isinf(results.skim_fields(fields)).all()
    assert np.isinf(results.skim_costs()).all()
    assert np.isinf(results.skim_turn_costs()).all()
    scratch = results.workspace.state_skims
    assert not scratch.flags.writeable
    assert not scratch.flags.owndata
    pointer = scratch.ctypes.data

    dijkstra(context, 0, None, results)
    np.testing.assert_array_equal(results.skim_fields(fields)[:, 0], [0, 2, 5])
    snapshot = scratch.copy()
    dijkstra(context, 0, 1, results)
    np.testing.assert_array_equal(results.skim_fields(fields)[:, 0], [0, 2, np.inf])
    assert results.workspace.state_skims.ctypes.data == pointer
    assert np.any(scratch != snapshot)
    assert_skims_match_paths(context, results, fields, results.skim_fields(fields))
    dijkstra(context, 2, 2, results)
    np.testing.assert_array_equal(results.skim_fields(fields)[:, 0], [np.inf, np.inf, 0])
    np.testing.assert_array_equal(results.skim_costs()[:, 0], [np.inf, np.inf, 0])
    np.testing.assert_array_equal(results.skim_turn_costs()[:, 0], [np.inf, np.inf, 0])

    old_snapshot = scratch.copy()
    results.skim_fields(fields * 2)
    assert results.workspace.skim_field_count == 2
    np.testing.assert_array_equal(scratch, old_snapshot)  # Resize pins old views.
    new_view = results.workspace.state_skims
    del results, context
    gc.collect()
    np.testing.assert_array_equal(scratch, old_snapshot)
    assert new_view.shape[1] == 2


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_edgeless_graph(context_type):
    context = context_type([0, 0, 0], [], [])
    results = dijkstra(context, 1, None)
    fields = [np.empty(0), np.empty(0)]
    skims = results.skim_fields(fields)
    np.testing.assert_array_equal(skims, [[np.inf, np.inf], [0, 0]])
    assert_skims_match_paths(context, results, fields, skims)


def test_attribute_ieee_values_do_not_change_routing():
    context = NodeBasedContext([0, 1, 2, 2], [1, 2], [1, 1])
    results = dijkstra(context, 0, None)
    fields = [np.array([np.nan, 1.]), np.array([np.inf, 2.]), np.array([-2., 1.])]
    np.testing.assert_array_equal(results.skim_fields(fields), [[0, 0, 0], [np.nan, np.inf, -2], [np.nan, np.inf, -1]])
    np.testing.assert_array_equal(results.skim_costs()[:, 0], [0, 1, 2])


def test_validation_and_read_only_inputs():
    context = history_context()
    results = dijkstra(context, 0, None)
    good = np.arange(context.link_count, dtype=np.float64)
    good.flags.writeable = False
    results.skim_fields([good])
    for bad in (good[:-1], good.reshape(2, 2), good.astype(np.float32), good.astype(np.int64), np.arange(8.)[::2]):
        with pytest.raises((TypeError, ValueError)):
            results.skim_fields([bad])
    unaligned = np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        results.skim_fields([unaligned])
    # A prior scratch column can be a contiguous input of link_count length,
    # but must not be read while the kernel overwrites the same scratch.
    node_context = NodeBasedContext([0, 1, 2], [1, 0], [1, 1])
    node_results = dijkstra(node_context, 0, None)
    node_results.skim_fields([np.ones(2)])
    with pytest.raises(ValueError, match="overlap"):
        node_results.skim_fields([node_results.workspace.state_skims[:, 0]])

    read_only = np.empty((5, 1))
    read_only.flags.writeable = False
    bad_outputs = [
        np.empty((5, 1), dtype=np.float32), np.empty((4, 1)), np.empty((5,)),
        np.empty((10, 1))[::2], read_only, [[0.]] * 5,
        np.ndarray((5, 1), dtype=np.float64, buffer=bytearray(41), offset=1),
    ]
    for bad in bad_outputs:
        for method in (lambda out: results.skim_fields([good], out), results.skim_costs, results.skim_turn_costs):
            with pytest.raises((TypeError, ValueError)):
                method(bad)
    with pytest.raises(ValueError, match="C-contiguous"):
        results.skim_fields([good, good], np.empty((5, 2), order="F"))
    with pytest.raises(ValueError, match="nonnegative"):
        results.workspace.prepare_skims(-1)
    with pytest.raises(TypeError):
        results.workspace.prepare_skims(1.5)
    workspace = AoNWorkspace(context, 2)
    assert workspace.state_skims.shape == (context.state_count, 2)
    assert workspace.context is context
    assert workspace.state_count == context.state_count


@pytest.mark.parametrize("turns", [False, True])
@pytest.mark.parametrize("seed", [17, 71, 123])
def test_random_multigraphs_against_path_sums(turns, seed):
    rng = np.random.default_rng(seed)
    n, m = 9, 40
    tails = np.sort(rng.integers(0, n - 1, m))
    heads = rng.integers(0, n - 1, m)  # Last node remains isolated.
    fs = np.r_[0, np.cumsum(np.bincount(tails, minlength=n))]
    costs = rng.integers(0, 6, m).astype(np.float64)
    if turns:
        turn_fs, to, penalties = [0], [], []
        for incoming in range(m):
            for outgoing in range(fs[heads[incoming]], fs[heads[incoming] + 1]):
                if rng.random() < 0.5:
                    to.append(outgoing)
                    penalties.append(rng.choice([0., 1., 4., np.inf]))
            turn_fs.append(len(to))
        context = TurnBasedContext(fs, heads, costs, turn_fs, to, penalties, allow_uturns=False)
    else:
        context = NodeBasedContext(fs, heads, costs)
    fields = [rng.normal(size=m) for _ in range(13)]
    results = context.make_results()
    for origin in range(n):
        for targets in (None, [origin], [0, n - 2]):
            dijkstra(context, origin, targets, results)
            assert_skims_match_paths(context, results, fields, results.skim_fields(fields))


@pytest.fixture(params=[None, 0.5, 10.])
def centroid_context(request):
    # Centroids: 0, 1, 2 (isolated). Paths to 1 use network nodes 3 and 4.
    # Node 3's cheapest arrival may differ from the one used on the path to 1.
    fs, heads, costs = [0, 2, 2, 2, 4, 5, 5], [3, 4, 1, 5, 3], [1, 1, 1, 100, 1]
    if request.param is None:
        return NodeBasedContext(fs, heads, costs)
    return TurnBasedContext(fs, heads, costs, [0, 1, 1, 1, 1, 1], [2], [request.param])


def test_centroid_od_direct_output(centroid_context):
    context = centroid_context
    z = 3
    fields = [np.array([10., 20., 30., 1000., 40.]), context.costs]
    cube = np.full((z, z, len(fields)), -123.)
    costs = np.full((z, z), -123.)
    turns = costs.copy()
    results = context.make_results()
    for origin in range(z):
        dijkstra(context, origin, range(z), results)
        out = cube[origin]
        assert results.skim_fields(fields, out=out, destination_count=z) is out
        cost_out = costs[origin].reshape(z, 1)
        assert results.skim_costs(out=cost_out, destination_count=z) is cost_out
        turn_out = turns[origin].reshape(z, 1)
        assert results.skim_turn_costs(out=turn_out, destination_count=z) is turn_out
        # Compare against an all-node search and output, not just the same limited call.
        full = dijkstra(context, origin, None)
        np.testing.assert_array_equal(out, full.skim_fields(fields)[:z])
        np.testing.assert_array_equal(cost_out, full.skim_costs()[:z])
        np.testing.assert_array_equal(turn_out, full.skim_turn_costs()[:z])
        assert results.workspace.state_skims.shape == (context.state_count, len(fields))
    np.testing.assert_array_equal(costs.diagonal(), np.zeros(z))
    np.testing.assert_array_equal(turns.diagonal(), np.zeros(z))
    assert np.isinf(cube[:2, 2]).all()
    assert np.isinf(cube[2, :2]).all()


@pytest.mark.parametrize("z", [0, 1, 2, 3, 6])
def test_centroid_output_bounds_and_partial_search(centroid_context, z):
    context = centroid_context
    results = dijkstra(context, 0, [0, 1])
    fields = [np.array([10., 20., 30., 1000., 40.]), context.costs]
    assert not results.reachable_to(5)  # Search stops before this network node settles.
    full = results.skim_fields(fields)
    scratch = results.workspace.state_skims
    pointer = scratch.ctypes.data
    snapshot = scratch.copy()
    # Guard rows check that the kernel only writes the requested number of rows.
    guarded = np.full((z + 2, len(fields)), -123.)
    out = guarded[1:-1]
    assert results.skim_fields(fields, out, destination_count=np.int64(z)) is out
    np.testing.assert_array_equal(out, full[:z])
    np.testing.assert_array_equal(guarded[[0, -1]], -123.)
    np.testing.assert_array_equal(results.skim_fields(fields, destination_count=z), full[:z])
    assert results.workspace.state_skims.ctypes.data == pointer
    np.testing.assert_array_equal(scratch, snapshot)
    for method in (results.skim_costs, results.skim_turn_costs):
        guarded = np.full((z + 2, 1), -123.)
        out = guarded[1:-1]
        assert method(out, destination_count=z) is out
        np.testing.assert_array_equal(out, method()[:z])
        np.testing.assert_array_equal(method(destination_count=z), method()[:z])
        np.testing.assert_array_equal(guarded[[0, -1]], -123.)
    assert results.skim_fields([], destination_count=z).shape == (z, 0)
    # The output count does not change the search's targets or settled states.
    assert results.destination_count == 2
    np.testing.assert_array_equal(results.destinations, [0, 1])


def test_centroid_output_before_search_and_origin_outside_prefix(centroid_context):
    context = centroid_context
    results = context.make_results()
    fields = [context.costs]
    assert np.isinf(results.skim_fields(fields, destination_count=2)).all()
    assert np.isinf(results.skim_costs(destination_count=2)).all()
    assert np.isinf(results.skim_turn_costs(destination_count=2)).all()
    dijkstra(context, 4, [1], results)
    np.testing.assert_array_equal(results.skim_fields(fields, destination_count=2)[:, 0], [np.inf, 2.])
    np.testing.assert_array_equal(results.skim_costs(destination_count=2)[:, 0], [np.inf, 2.])
    np.testing.assert_array_equal(results.skim_turn_costs(destination_count=2)[:, 0], [np.inf, 0.])


def test_centroid_count_validation(centroid_context):
    context = centroid_context
    results = dijkstra(context, 0, None)
    methods = (
        lambda **kwargs: results.skim_fields([context.costs], **kwargs),
        results.skim_costs, results.skim_turn_costs,
    )
    for method in methods:
        for count in (-1, context.node_count + 1, 2**100):
            with pytest.raises(ValueError, match="destination_count"):
                method(destination_count=count)
        for count in (1.5, "2", [2]):
            with pytest.raises(TypeError):
                method(destination_count=count)
        with pytest.raises(ValueError, match="shape"):
            method(out=np.empty((context.node_count, 1)), destination_count=2)
        with pytest.raises(ValueError, match="shape"):
            method(out=np.empty((2, 1)))  # No implicit count from output shape.
        np.testing.assert_array_equal(method(destination_count=None), method())


def test_prepared_fields_own_od_outputs_without_copying_inputs(centroid_context):
    context = centroid_context
    source = np.array([10., 20., 30., 1000., 40.])
    prepared = SkimmingContext(context, [source, context.costs], 3, include_costs=True, include_turn_costs=True)
    assert prepared.context is context
    assert prepared.centroid_count == 3
    assert prepared.field_count == 2
    assert prepared.fields[0].ctypes.data == source.ctypes.data
    assert np.shares_memory(prepared.fields[0], source)
    assert not source.flags.writeable
    assert prepared.od_skims.shape == (3, 3, 2)
    assert prepared.od_costs.shape == prepared.od_turn_costs.shape == (3, 3)
    arrays = (prepared.od_skims, prepared.od_costs, prepared.od_turn_costs)
    for array in (*prepared.fields, *arrays):
        assert not array.flags.writeable
        assert not array.flags.owndata
        assert array.flags.c_contiguous
        with pytest.raises(ValueError):
            array.flags.writeable = True
    for array in arrays:
        assert np.isinf(array).all()
    pointers = [array.ctypes.data for array in arrays]
    results = context.make_results()
    results.prepare_skims(prepared)  # Allowed before any search.
    scratch_pointer = results.workspace.state_skims.ctypes.data
    for origin in range(3):
        dijkstra(context, origin, range(3), results)
        row = results.skim_fields(prepared)
        assert np.shares_memory(row, prepared.od_skims)
        assert not row.flags.writeable
        np.testing.assert_array_equal(row, results.skim_fields([source, context.costs], destination_count=3))
        np.testing.assert_array_equal(prepared.od_costs[origin], results.skim_costs(destination_count=3)[:, 0])
        np.testing.assert_array_equal(
            prepared.od_turn_costs[origin], results.skim_turn_costs(destination_count=3)[:, 0]
        )
        assert results.workspace.state_skims.ctypes.data == scratch_pointer
    assert [array.ctypes.data for array in arrays] == pointers
    assert np.isinf(prepared.od_skims[0, 2]).all()
    assert np.isinf(prepared.od_skims[2, :2]).all()

    # A later partial search must overwrite old finite entries in its row only.
    snapshot = prepared.od_skims.copy()
    dijkstra(context, 0, 0, results)
    results.skim_fields(prepared)
    np.testing.assert_array_equal(arrays[0][0, 0], [0., 0.])
    assert np.isinf(arrays[0][0, 1:]).all()
    assert np.isinf(arrays[1][0, 1:]).all()
    assert np.isinf(arrays[2][0, 1:]).all()
    np.testing.assert_array_equal(arrays[0][1:], snapshot[1:])
    assert np.isfinite(snapshot[0, 1]).all()


def test_prepared_calls_do_not_rebuild_inputs_or_scratch(monkeypatch):
    context = history_context()
    source = np.arange(context.link_count, dtype=np.float64)
    prepared = SkimmingContext(context, (array for array in [source]))
    results = dijkstra(context, 0, None)
    results.prepare_skims(prepared)
    assert prepared.od_costs is prepared.od_turn_costs is None

    def unexpected(*args, **kwargs):
        raise AssertionError("prepared calls must reuse inputs and scratch")

    scratch = results.workspace.state_skims
    with monkeypatch.context() as patch:
        # Returning read-only NumPy views is allowed; rebuilding inputs is not.
        patch.setattr(np, "array", unexpected)
        patch.setattr(np, "shares_memory", unexpected)
        patch.setattr(np, "full", unexpected)
        for _ in range(100):
            results.skim_fields(prepared)
    assert np.shares_memory(scratch, results.workspace.state_skims)
    np.testing.assert_array_equal(prepared.od_skims[0], results.skim_fields([source]))
    # Switching widths replaces scratch; returning to prepared input must prepare it again.
    results.skim_fields([source, source])
    results.skim_fields(prepared)
    assert results.workspace.state_skims.shape == (context.state_count, 1)
    np.testing.assert_array_equal(prepared.od_skims[0], results.skim_fields([source]))


@pytest.mark.parametrize(
    "include_costs,include_turn_costs", [(False, False), (True, False), (False, True), (True, True)]
)
def test_prepared_cost_only_and_empty_outputs(include_costs, include_turn_costs):
    context = history_context(0.5)
    prepared = SkimmingContext(context, [], 3, include_costs=include_costs, include_turn_costs=include_turn_costs)
    results = dijkstra(context, 0, range(3))
    assert results.skim_fields(prepared).shape == (3, 0)
    assert prepared.od_skims.shape == (3, 3, 0)
    if include_costs:
        np.testing.assert_array_equal(prepared.od_costs[0], results.skim_costs(destination_count=3)[:, 0])
    else:
        assert prepared.od_costs is None
    if include_turn_costs:
        np.testing.assert_array_equal(prepared.od_turn_costs[0], results.skim_turn_costs(destination_count=3)[:, 0])
    else:
        assert prepared.od_turn_costs is None
    empty = SkimmingContext(context, [context.costs], 0, include_costs=True, include_turn_costs=True)
    assert empty.od_skims.shape == (0, 0, 1)
    assert empty.od_costs.shape == empty.od_turn_costs.shape == (0, 0)
    with pytest.raises(ValueError, match="origin"):
        results.skim_fields(empty)


@pytest.mark.parametrize("context_type", [NodeBasedContext, TurnBasedContext])
def test_prepared_edgeless_graph(context_type):
    context = context_type([0, 0, 0], [], [])
    prepared = SkimmingContext(context, [np.empty(0)], include_costs=True, include_turn_costs=True)
    results = dijkstra(context, 1, None)
    np.testing.assert_array_equal(results.skim_fields(prepared)[:, 0], [np.inf, 0.])
    np.testing.assert_array_equal(prepared.od_costs[1], [np.inf, 0.])
    np.testing.assert_array_equal(prepared.od_turn_costs[1], [np.inf, 0.])


def test_prepared_ownership_and_retained_views():
    context = history_context()
    source = np.arange(context.link_count, dtype=np.float64)
    source_ref = weakref.ref(source)
    prepared = SkimmingContext(context, [source], include_costs=True, include_turn_costs=True)
    del source
    gc.collect()
    assert source_ref() is not None  # The pointer table alone would not retain the input.
    results = dijkstra(context, 0, None)
    results.skim_fields(prepared)
    input_view = prepared.fields[0]
    views = (prepared.od_skims, prepared.od_costs, prepared.od_turn_costs)
    snapshots = [array.copy() for array in views]
    del prepared, results, context
    gc.collect()
    np.testing.assert_array_equal(input_view, np.arange(4.))
    for array, snapshot in zip(views, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_prepared_validation_and_workspace_alias():
    context = history_context()
    good = np.arange(context.link_count, dtype=np.float64)
    for bad in (good[:-1], good.astype(np.float32), good.astype(np.int64), good.reshape(2, 2), np.arange(8.)[::2]):
        with pytest.raises((TypeError, ValueError)):
            SkimmingContext(context, [bad])
    with pytest.raises(TypeError, match="NumPy"):
        SkimmingContext(context, [[1., 2., 3., 4.]])
    unaligned = np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        SkimmingContext(context, [unaligned])
    for count in (-1, context.node_count + 1, 2**100):
        with pytest.raises(ValueError, match="centroid_count"):
            SkimmingContext(context, [good], count)
    for count in (1.5, "2"):
        with pytest.raises(TypeError):
            SkimmingContext(context, [good], count)
    with pytest.raises(TypeError, match="GraphContext"):
        SkimmingContext(None, [good])
    assert good.flags.writeable  # Failed construction did not change valid inputs.
    prepared = SkimmingContext(context, [good], np.int64(3))
    with pytest.raises(RuntimeError, match="reinitialized"):
        prepared.__init__(context, [good])
    results = context.make_results()
    with pytest.raises(ValueError, match="search"):
        results.skim_fields(prepared)
    with pytest.raises(TypeError, match="SkimmingContext"):
        results.prepare_skims(None)
    dijkstra(context, 0, range(3), results)
    for kwargs in ({"out": np.empty((3, 1))}, {"destination_count": 3}):
        with pytest.raises(ValueError, match="already supplies"):
            results.skim_fields(prepared, **kwargs)
    with pytest.raises(ValueError, match="different context"):
        dijkstra(history_context(), 0, None).skim_fields(prepared)
    dijkstra(context, 3, None, results)
    with pytest.raises(ValueError, match="origin"):
        results.skim_fields(prepared)

    # Borrowing a workspace column is unsafe: summation overwrites its own input.
    context = NodeBasedContext([0, 1, 2], [1, 0], [1, 1])
    results = dijkstra(context, 0, None)
    results.skim_fields([context.costs])
    prepared = SkimmingContext(context, [results.workspace.state_skims[:, 0]])
    with pytest.raises(ValueError, match="overlap"):
        results.prepare_skims(prepared)
    with pytest.raises(ValueError, match="overlap"):
        results.skim_fields(prepared)


def test_prepared_fields_shared_across_worker_origins(centroid_context):
    context = centroid_context
    fields = [np.arange(context.link_count, dtype=np.float64), context.costs]
    prepared = SkimmingContext(context, fields, 3, include_costs=True, include_turn_costs=True)
    expected = []
    for origin in range(3):
        results = dijkstra(context, origin, range(3))
        expected.append((results.skim_fields(fields, destination_count=3), results.skim_costs(destination_count=3),
                         results.skim_turn_costs(destination_count=3)))

    def run(origin):
        results = context.make_results()
        results.prepare_skims(prepared)
        for _ in range(20):
            dijkstra(context, origin, range(3), results)
            results.skim_fields(prepared)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, range(3)))
    for origin, (skims, costs, turns) in enumerate(expected):
        np.testing.assert_array_equal(prepared.od_skims[origin], skims)
        np.testing.assert_array_equal(prepared.od_costs[origin], costs[:, 0])
        np.testing.assert_array_equal(prepared.od_turn_costs[origin], turns[:, 0])


def test_shared_context_separate_worker_workspaces():
    context = history_context(1)
    fields = [np.arange(context.link_count, dtype=np.float64), context.costs]

    def run(origin):
        results = dijkstra(context, origin, None)
        return results.skim_fields(fields), results.skim_costs(), results.skim_turn_costs()

    origins = list(range(context.node_count)) * 8
    expected = [run(origin) for origin in origins]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(run, origins))
    for a, e in zip(actual, expected, strict=True):
        for aa, ee in zip(a, e, strict=True):
            np.testing.assert_array_equal(aa, ee)
