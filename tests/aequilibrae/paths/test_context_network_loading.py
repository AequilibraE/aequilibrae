"""Standalone loading and assignment share small, independent buffer owners."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.outputs import LoadingOutputs
from aequilibrae.paths.cython.network_loading import network_loading, reduce_loading_outputs
from aequilibrae.paths.cython.workspaces import AoNWorkspace, LoadingWorkspace, SkimmingWorkspace, SelectLinkWorkspace
from .routing_helpers import allocate_results, history_context, make_context, path_walk_outputs, search


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("penalty", [0.5, 10.0, np.inf])
@pytest.mark.parametrize("cores", [1, 3])
def test_loading_and_turn_totals_against_path_walks(turn, penalty, cores):
    context = history_context(penalty, turn=turn)
    demand = np.arange(48, dtype=np.float64).reshape(4, 4, 3) - 10
    prepared = PreparedAoN(context, demand, cores=cores)
    out = prepared.make_outputs()
    expected, _, total, _, _ = path_walk_outputs(context, demand)
    for _ in range(3):
        assert prepared.run(out) is out
        np.testing.assert_allclose(out.loading.link_loads, expected)
        assert out.turn_cost_total == total


def test_fixed_demand_is_retained_without_changing_writeability():
    context = history_context()
    demand = np.ones((4, 4, 1))
    reference = weakref.ref(demand)
    prepared = PreparedAoN(context, demand)
    expected = path_walk_outputs(context, demand)[0]
    assert demand.flags.writeable
    del demand
    gc.collect()
    assert reference() is not None
    out = prepared.make_outputs()
    for _ in range(3):
        prepared.run(out)
        np.testing.assert_array_equal(out.loading.link_loads, expected)
    del prepared
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("turn", [False, True])
def test_edgeless_network_and_intrazonal_demand(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    demand = np.ones((2, 2, 1))
    prepared = PreparedAoN(context, demand, cores=3)
    out = prepared.run(prepared.make_outputs())
    assert out.loading.link_loads.shape == (0, 1)
    assert out.turn_cost_total == 0


@pytest.mark.parametrize("turn", [False, True])
def test_demand_classes_do_not_cancel_target_selection(turn):
    context = history_context(turn=turn)
    demand = np.zeros((4, 4, 2))
    demand[0, 3] = [1, -1]
    prepared = PreparedAoN(context, demand)
    out = prepared.run(prepared.make_outputs())
    expected = path_walk_outputs(context, demand)[0]
    assert np.any(expected)
    np.testing.assert_array_equal(out.loading.link_loads, expected)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("classes", [0, 1, 3, 37])
def test_standalone_loading_reuses_paths_scratch_and_output(turn, classes):
    context = history_context(turn=turn)
    results = allocate_results(context)
    workspace = LoadingWorkspace(context.state_count, classes)
    output = LoadingOutputs(context.link_count, classes)
    demand = np.arange(16 * classes, dtype=np.float64).reshape(4, 4, classes) - 3
    queries = [LoadingQuery(row) for row in demand]
    expected = path_walk_outputs(context, demand)[0]
    scratch_view, output_view = workspace.state_loads, output.link_loads
    for _ in range(3):
        output.reset()
        for origin, query in enumerate(queries):
            search(context, origin, results=results)
            before = results.predecessors.copy(), results.distances.copy(), results.settled_count
            assert network_loading(results, query, workspace, output) is output
            np.testing.assert_array_equal(results.predecessors, before[0])
            np.testing.assert_array_equal(results.distances, before[1])
            assert results.settled_count == before[2]
        np.testing.assert_allclose(output.link_loads, expected)
        assert scratch_view.ctypes.data == workspace.state_loads.ctypes.data
        assert output_view.ctypes.data == output.link_loads.ctypes.data
    output.reset()
    assert not np.any(output_view)
    # Reset clears output, not the independent cascade scratch.
    assert scratch_view.ctypes.data == workspace.state_loads.ctypes.data


@pytest.mark.parametrize("turn", [False, True])
def test_before_search_partial_search_and_empty_demand(turn):
    context = history_context(turn=turn)
    results = allocate_results(context)
    workspace = LoadingWorkspace(context.state_count, 1)
    output = LoadingOutputs(context.link_count, 1)
    query = LoadingQuery(np.array([[100.0], [2.0], [3.0], [4.0]]))
    network_loading(results, query, workspace, output)
    assert not np.any(output.link_loads)
    assert not np.any(workspace.state_loads)

    search(context, 0, 1, results)
    network_loading(results, query, workspace, output)
    np.testing.assert_array_equal(output.link_loads[:, 0], [2, 0, 0, 0])
    assert workspace.state_loads[results.root, 0] == 2
    assert not results.reachable_to(3)  # Loading did not extend the search.
    network_loading(results, LoadingQuery(np.empty((0, 1))), workspace, output)
    np.testing.assert_array_equal(output.link_loads[:, 0], [2, 0, 0, 0])
    assert not np.any(workspace.state_loads)  # Empty queries still replace scratch.


@pytest.mark.parametrize("turn", [False, True])
def test_destination_prefix_can_use_states_outside_output_rows(turn):
    context = make_context([0, 1, 1, 2], [2, 1], [1, 1], turn=turn)
    results = search(context, 0)
    workspace = LoadingWorkspace(context.state_count, 1)
    output = LoadingOutputs(2, 1)
    network_loading(results, LoadingQuery(np.array([[7.0], [5.0]])), workspace, output)
    np.testing.assert_array_equal(output.link_loads[:, 0], [5, 5])


@pytest.mark.parametrize("turn", [False, True])
def test_standalone_edgeless_and_nonfinite_demand(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    results = search(context, 0)
    workspace = LoadingWorkspace(context.state_count, 1)
    output = LoadingOutputs(0, 1)
    network_loading(results, LoadingQuery(np.array([[np.nan], [np.inf]])), workspace, output)
    assert output.link_loads.shape == (0, 1)
    assert not np.any(workspace.state_loads)
    output.reset()

    branches = make_context([0, 2, 2, 2], [1, 2], [1, 1], turn=turn)
    results = search(branches, 0)
    workspace = LoadingWorkspace(branches.state_count, 1)
    output = LoadingOutputs(2, 1)
    network_loading(results, LoadingQuery(np.array([[np.inf], [np.nan], [-3.0]])), workspace, output)
    assert np.isnan(output.link_loads[0, 0])
    assert output.link_loads[1, 0] == -3  # NaN does not spread to a sibling path.


def test_query_borrows_demand_without_changing_writeability():
    demand = np.ones((4, 2))
    query = LoadingQuery(demand)
    assert query.destination_count == 4 and query.class_count == 2
    assert np.shares_memory(query.demand, demand)
    assert demand.flags.writeable
    demand *= 3
    np.testing.assert_array_equal(query.demand, demand)
    demand.flags.writeable = False
    readonly = LoadingQuery(demand)
    assert np.shares_memory(readonly.demand, demand)
    with pytest.raises(ValueError):
        query.demand.flags.writeable = True


def test_owners_have_independent_lifetimes():
    context = history_context()
    results = search(context, 0)
    group = AoNWorkspace(context.state_count, class_count=1, field_count=2, select_links=True)
    workspace = group.loading
    output = LoadingOutputs(context.link_count, 1)
    demand = np.ones((context.node_count, 1))
    demand_ref = weakref.ref(demand)
    query = LoadingQuery(demand)
    del context, group, demand
    gc.collect()
    assert demand_ref() is not None
    network_loading(results, query, workspace, output)
    del query, results
    gc.collect()
    assert demand_ref() is None  # Neither scratch nor output retained the inputs.

    scratch_view, output_view = workspace.state_loads, output.link_loads
    snapshots = scratch_view.copy(), output_view.copy()
    scratch_owner, output_owner = weakref.ref(scratch_view.base), weakref.ref(output_view.base)
    del workspace, output
    gc.collect()
    np.testing.assert_array_equal(scratch_view, snapshots[0])
    np.testing.assert_array_equal(output_view, snapshots[1])
    assert scratch_owner() is not None and output_owner() is not None
    del scratch_view, output_view
    gc.collect()
    assert scratch_owner() is None and output_owner() is None


def test_workspace_group_allocates_only_requested_operations():
    group = AoNWorkspace(5)
    assert group.loading is group.skimming is group.select_link is None
    group = AoNWorkspace(5, class_count=2)
    assert isinstance(group.loading, LoadingWorkspace)
    assert group.skimming is group.select_link is None
    full = AoNWorkspace(5, class_count=2, field_count=3, select_links=True)
    assert isinstance(full.skimming, SkimmingWorkspace)
    assert isinstance(full.select_link, SelectLinkWorkspace)
    assert full.loading.state_loads.shape == (5, 2)
    assert full.skimming.state_skims.shape == (5, 3)
    assert np.all(np.isinf(full.skimming.state_skims))
    assert not np.any(full.select_link.selected_paths)
    assert SkimmingWorkspace(5, 0).state_skims.shape == (5, 0)
    assert LoadingWorkspace(5, 0).state_loads.shape == (5, 0)
    for owner in (full, full.loading, full.skimming, full.select_link):
        assert not hasattr(owner, "context")
        for method in ("prepare_loading", "prepare_skims", "prepare_select_links"):
            assert not hasattr(owner, method)
    for values in (full.loading.state_loads, full.skimming.state_skims, full.select_link.selected_paths):
        with pytest.raises(ValueError):
            values.flat[0] = 0
        with pytest.raises(ValueError):
            values.flags.writeable = True


@pytest.mark.parametrize(
    "factory, args",
    [
        (LoadingWorkspace, (5, 2)),
        (SkimmingWorkspace, (5, 3)),
        (SelectLinkWorkspace, (5,)),
        (AoNWorkspace, (5,)),
        (LoadingOutputs, (0, 0)),
        (LoadingQuery, (np.empty((0, 0)),)),
    ],
)
def test_fixed_layout_cannot_be_reinitialized(factory, args):
    owner = factory(*args)
    with pytest.raises(RuntimeError):
        owner.__init__(*args)


@pytest.mark.parametrize(
    "factory, args",
    [
        (LoadingWorkspace, (0, 2)),
        (LoadingWorkspace, (5, -1)),
        (SkimmingWorkspace, (0, 2)),
        (SkimmingWorkspace, (5, -1)),
        (SelectLinkWorkspace, (0,)),
        (AoNWorkspace, (0,)),
        (LoadingOutputs, (-1, 0)),
        (LoadingOutputs, (0, -1)),
    ],
)
def test_invalid_dimensions(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [[1.0]],
        np.ones(3),
        np.ones((2, 3), dtype=np.float32),
        np.ones((4, 3))[::2],
        np.ones((2, 6))[:, ::2],
        np.ndarray((2, 3), dtype=np.float64, buffer=bytearray(49), offset=1),
    ],
)
def test_invalid_demand_layout(bad):
    with pytest.raises((TypeError, ValueError)):
        LoadingQuery(bad)


def test_loading_validates_dimensions_before_any_writes():
    context = history_context()
    results = search(context, 0)
    query = LoadingQuery(np.ones((4, 2)))
    workspace = LoadingWorkspace(context.state_count, 2)
    output = LoadingOutputs(context.link_count, 2)
    network_loading(results, query, workspace, output)
    before = output.link_loads.copy(), workspace.state_loads.copy()
    bad_calls = [
        (LoadingQuery(np.ones((5, 2))), workspace, output),
        (query, LoadingWorkspace(context.state_count + 1, 2), output),
        (query, LoadingWorkspace(context.state_count, 1), output),
        (query, workspace, LoadingOutputs(context.link_count + 1, 2)),
        (query, workspace, LoadingOutputs(context.link_count, 1)),
    ]
    for bad_query, bad_workspace, bad_output in bad_calls:
        with pytest.raises(ValueError):
            network_loading(results, bad_query, bad_workspace, bad_output)
        np.testing.assert_array_equal(output.link_loads, before[0])
        np.testing.assert_array_equal(workspace.state_loads, before[1])


@pytest.mark.parametrize("turn", [False, True])
def test_worker_outputs_reduce_without_changing_workers(turn):
    context = history_context(turn=turn)
    demand = np.arange(32, dtype=np.float64).reshape(4, 4, 2)

    def run(origins):
        results = allocate_results(context)
        workspace = LoadingWorkspace(context.state_count, 2)
        output = LoadingOutputs(context.link_count, 2)
        for origin in origins:
            search(context, origin, results=results)
            network_loading(results, LoadingQuery(demand[origin]), workspace, output)
        return output

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = list(pool.map(run, ([0, 2], [1, 3])))
    snapshots = [worker.link_loads.copy() for worker in workers]
    output = LoadingOutputs(context.link_count, 2)
    retained = output.link_loads
    expected = path_walk_outputs(context, demand)[0]
    for _ in range(3):
        assert reduce_loading_outputs(iter(workers), output) is output
        np.testing.assert_array_equal(retained, expected)
        for worker, snapshot in zip(workers, snapshots, strict=True):
            np.testing.assert_array_equal(worker.link_loads, snapshot)
    reduce_loading_outputs([], output)
    assert not np.any(retained)
    assert retained.ctypes.data == output.link_loads.ctypes.data


def test_reduction_validation_precedes_reset():
    context = history_context()
    output = LoadingOutputs(context.link_count, 1)
    network_loading(search(context, 0), LoadingQuery(np.ones((4, 1))), LoadingWorkspace(context.state_count, 1), output)
    before = output.link_loads.copy()
    for workers in ([output], [LoadingOutputs(3, 1)], [LoadingOutputs(4, 2)], [None], [object()]):
        with pytest.raises((TypeError, ValueError)):
            reduce_loading_outputs(workers, output)
        np.testing.assert_array_equal(output.link_loads, before)
    for shape in ((0, 2), (2, 0), (0, 0)):
        empty_output = LoadingOutputs(*shape)
        reduce_loading_outputs([LoadingOutputs(*shape)], empty_output)
        assert empty_output.link_loads.shape == shape
