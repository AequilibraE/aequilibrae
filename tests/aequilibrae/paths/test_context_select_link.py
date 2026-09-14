"""Select-link inputs, scratch and optional outputs work without assignment."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import AoNOutputs, PreparedAoN
from aequilibrae.paths.cython.network_loading import network_loading
from aequilibrae.paths.cython.outputs import (
    LoadingOutputs, SelectLinkLoadingOutputs, SelectLinkODOutputs, SelectLinkOutputs,
)
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.context import SelectLinkContext
from aequilibrae.paths.cython.select_link_loading import (
    select_link_loading, reduce_select_link_loading_outputs,
)
from aequilibrae.paths.cython.workspaces import LoadingWorkspace, SelectLinkWorkspace
from .routing_helpers import allocate_results, history_context, make_context, path_walk_outputs, search


class TrackedSelectLinkContext(SelectLinkContext):
    """A Python subclass lets lifetime tests weak-reference the input owner."""


def walk_selected_paths(results, demand, selections):
    """Walk each available path rather than using the membership/cascade passes."""
    classes = demand.shape[1]
    loads = np.zeros((len(selections), results.link_count, classes))
    od = np.zeros((len(selections), len(demand), classes))
    for node in range(len(demand)):
        links = results.path_links_to(node)
        for index, members in enumerate(selections.values()):
            if any(link in members for link in links):
                od[index, node] = demand[node]
                for link in links:
                    loads[index, link] += demand[node]
    return loads, od


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("classes", [0, 1, 3, 37])
@pytest.mark.parametrize("link_loads, od", [(True, True), (True, False), (False, True), (False, False)])
def test_optional_outputs_reuse_buffers_and_match_path_walks(turn, classes, link_loads, od):
    context = history_context(turn=turn)
    selections = {"first": [0, 3, 3], "last": [2], "overlap": [2, 3], "empty": []}
    inputs = SelectLinkContext(context.link_count, selections)
    output = inputs.make_outputs(4, classes, origin_count=2, link_loads=link_loads, od=od)
    assert (output.loading is not None) == link_loads
    assert (output.od is not None) == od
    flags = SelectLinkWorkspace(context.state_count)
    loading = LoadingWorkspace(context.state_count, classes) if link_loads else None
    results = allocate_results(context)
    demand = np.arange(4 * classes, dtype=np.float64).reshape(4, classes) - 4
    query = LoadingQuery(demand)
    retained = [flags.selected_paths]
    if loading is not None:
        retained += [loading.state_loads, output.loading.link_loads]
    if od:
        retained.append(output.od.demand)
    pointers = [view.ctypes.data for view in retained]
    accumulated = np.zeros((len(selections), context.link_count, classes))
    # Borrowed demand changes, but neither queries nor output need reallocation.
    for origin in (0, 2, 1, 0):
        demand += 1
        search(context, origin, results=results)
        old_paths = results.predecessors.copy(), results.distances.copy(), results.settled_count
        if od:
            other_row = output.od.demand[0].copy()
        assert select_link_loading(results, query, inputs, flags, loading,
                                   output.loading, output.od, origin_row=1) == (output.loading, output.od)
        expected_loads, expected_od = walk_selected_paths(results, demand, selections)
        accumulated += expected_loads
        if link_loads:
            np.testing.assert_allclose(output.loading.link_loads, accumulated)
        if od:
            np.testing.assert_array_equal(output.od.demand[1], expected_od)
            np.testing.assert_array_equal(output.od.demand[0], other_row)
        np.testing.assert_array_equal(results.predecessors, old_paths[0])
        np.testing.assert_array_equal(results.distances, old_paths[1])
        assert results.settled_count == old_paths[2]
        current = [flags.selected_paths]
        if link_loads:
            current += [loading.state_loads, output.loading.link_loads]
        if od:
            current.append(output.od.demand)
        assert [view.ctypes.data for view in current] == pointers
        for old_view, current_view in zip(retained, current, strict=True):
            np.testing.assert_array_equal(old_view, current_view)
    output.reset()
    if link_loads:
        assert not np.any(output.loading.link_loads)
        assert pointers[1:3] == [loading.state_loads.ctypes.data, output.loading.link_loads.ctypes.data]
    if od:
        assert not np.any(output.od.demand)
        assert pointers[-1] == output.od.demand.ctypes.data


@pytest.mark.parametrize("turn", [False, True])
def test_partial_presearch_and_empty_queries_replace_scratch_and_od(turn):
    context = history_context(turn=turn)
    inputs = SelectLinkContext(4, {"all": range(4)})
    output = inputs.make_outputs(4, 1)
    flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 1)
    results = allocate_results(context)
    query = LoadingQuery(np.array([[100.], [2.], [3.], [4.]]))
    # Fill buffers, then use a fresh pre-search result to expose stale data.
    select_link_loading(search(context, 0), query, inputs, flags, loading, output.loading, output.od)
    saved_loads = output.loading.link_loads.copy()
    select_link_loading(results, query, inputs, flags, loading, output.loading, output.od)
    assert not np.any(flags.selected_paths)
    assert not np.any(loading.state_loads)
    assert not np.any(output.od.demand)
    np.testing.assert_array_equal(output.loading.link_loads, saved_loads)

    output.reset()
    search(context, 0, 1, results)
    select_link_loading(results, query, inputs, flags, loading, output.loading, output.od)
    np.testing.assert_array_equal(output.loading.link_loads[0, :, 0], [2, 0, 0, 0])
    np.testing.assert_array_equal(output.od.demand[0, 0, :, 0], [0, 2, 0, 0])
    assert not results.reachable_to(3)
    finalized = results.settlement_order[:results.settled_count]
    expected_flags = np.zeros(context.state_count, dtype=bool)
    expected_flags[finalized[1:]] = True
    np.testing.assert_array_equal(flags.selected_paths, expected_flags)
    assert loading.state_loads[results.root, 0] == 2

    empty_od = inputs.make_outputs(0, 1, link_loads=False).od
    select_link_loading(results, LoadingQuery(np.empty((0, 1))), inputs, flags, loading,
                        output.loading, empty_od)
    assert not np.any(loading.state_loads)
    np.testing.assert_array_equal(output.loading.link_loads[0, :, 0], [2, 0, 0, 0])
    np.testing.assert_array_equal(flags.selected_paths, expected_flags)


@pytest.mark.parametrize("turn", [False, True])
def test_selected_paths_outside_destination_prefix_and_full_ancestor_loading(turn):
    context = make_context([0, 1, 1, 2], [2, 1], [1, 1], turn=turn)
    inputs = SelectLinkContext(2, {"last": [1]})
    output = inputs.make_outputs(2, 1)
    select_link_loading(search(context, 0), LoadingQuery(np.array([[100.], [5.]])), inputs,
                        SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 1),
                        output.loading, output.od)
    np.testing.assert_array_equal(output.loading.link_loads[0, :, 0], [5, 5])
    np.testing.assert_array_equal(output.od.demand[0, 0, :, 0], [0, 5])


def test_membership_uses_turn_state_history_not_intermediate_terminal():
    context = history_context()
    results = search(context, 0)
    inputs = SelectLinkContext(4, {"cheap_arrival": [0], "through_arrival": [3]})
    output = inputs.make_outputs(4, 1)
    flags = SelectLinkWorkspace(context.state_count)
    select_link_loading(results, LoadingQuery(np.ones((4, 1))), inputs, flags,
                        LoadingWorkspace(context.state_count, 1), output.loading, output.od)
    np.testing.assert_array_equal(output.od.demand[0, :, :, 0], [[0, 1, 0, 0], [0, 0, 0, 1]])
    assert not flags.selected_paths[results.terminal_states[1]]
    assert flags.selected_paths[results.terminal_states[3]]
    np.testing.assert_array_equal(output.loading.link_loads[1, :, 0], [0, 1, 1, 1])


@pytest.mark.parametrize("turn", [False, True])
def test_zero_cost_cycles_and_nonfinite_demand(turn):
    context = make_context([0, 2, 4, 5, 5], [1, 3, 0, 2, 1], [0, 1, 0, 0, 0], turn=turn)
    inputs = SelectLinkContext(5, {"cycle": [0, 2, 3, 4], "branch": [1], "empty": []})
    results = search(context, 0)
    query = LoadingQuery(np.array([[np.nan, np.inf], [np.nan, -3], [np.inf, 2], [4, -np.inf]]))
    output = inputs.make_outputs(4, 2)
    select_link_loading(results, query, inputs, SelectLinkWorkspace(context.state_count),
                        LoadingWorkspace(context.state_count, 2), output.loading, output.od)
    expected = walk_selected_paths(results, query.demand, {"cycle": [0, 2, 3, 4], "branch": [1], "empty": []})
    np.testing.assert_array_equal(output.loading.link_loads, expected[0])
    np.testing.assert_array_equal(output.od.demand[0], expected[1])
    assert not np.any(output.loading.link_loads[2])


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("classes", [0, 1])
@pytest.mark.parametrize("selections", [{}, {"empty": []}])
def test_empty_links_sets_classes_and_origin_rows(turn, classes, selections):
    context = make_context([0, 0, 0], [], [], turn=turn)
    inputs = SelectLinkContext(0, selections)
    output = inputs.make_outputs(2, classes)
    select_link_loading(search(context, 0), LoadingQuery(np.ones((2, classes))), inputs,
                        SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, classes),
                        output.loading, output.od)
    assert output.loading.link_loads.shape == (len(selections), 0, classes)
    assert output.od.demand.shape == (1, len(selections), 2, classes)
    assert not np.any(output.od.demand)
    output.reset()
    empty_rows = inputs.make_outputs(2, classes, origin_count=0)
    empty_rows.reset()
    with pytest.raises(ValueError, match="origin_row"):
        select_link_loading(search(context, 0), LoadingQuery(np.ones((2, classes))), inputs,
                            SelectLinkWorkspace(context.state_count), od_output=empty_rows.od)
    reduce_select_link_loading_outputs([output.loading], empty_rows.loading)


def test_od_only_does_not_touch_loading_scratch_and_zero_classes_still_marks_paths():
    context = history_context()
    results = search(context, 0)
    inputs = SelectLinkContext(4, {"all": range(4)})
    flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 1)
    query = LoadingQuery(np.ones((4, 1)))
    network_loading(results, query, loading, LoadingOutputs(4, 1))
    before = loading.state_loads.copy()
    select_link_loading(results, query, inputs, flags, loading, od_output=inputs.make_outputs(4, 1).od)
    np.testing.assert_array_equal(loading.state_loads, before)
    zero_classes = inputs.make_outputs(4, 0)
    select_link_loading(results, LoadingQuery(np.empty((4, 0))), inputs, flags,
                        LoadingWorkspace(context.state_count, 0), zero_classes.loading, zero_classes.od)
    assert np.any(flags.selected_paths)
    select_link_loading(allocate_results(context), LoadingQuery(np.empty((4, 0))), inputs, flags,
                        LoadingWorkspace(context.state_count, 0), zero_classes.loading, zero_classes.od)
    assert not np.any(flags.selected_paths)


def test_owned_masks_named_views_and_independent_lifetimes():
    members = np.array([0, 3, 3])
    members_ref = weakref.ref(members)
    inputs = TrackedSelectLinkContext(4, {"first": members, "second": [2]})
    assert inputs.set_names == ("first", "second")
    assert inputs.set_count == 2 and inputs.link_count == 4
    assert members.flags.writeable
    members[:] = 1
    np.testing.assert_array_equal(inputs.masks, [[True, False, False, True], [False, False, True, False]])
    del members
    gc.collect()
    assert members_ref() is None
    input_ref = weakref.ref(inputs)
    output = inputs.make_outputs(4, 2, origin_count=3)
    loading, od = output.loading, output.od
    del output, inputs
    gc.collect()
    assert input_ref() is None  # Outputs did not retain selection inputs.
    inputs = SelectLinkContext(4, {"first": [0, 3], "second": [2]})
    context = history_context()
    select_link_loading(search(context, 0), LoadingQuery(np.ones((4, 2))), inputs,
                        SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 2),
                        loading, od, origin_row=2)
    loads, matrices = loading.loads, od.matrices
    assert list(loads) == list(matrices) == ["first", "second"]
    assert loading.link_loads.flags.c_contiguous and od.demand.flags.c_contiguous
    assert not matrices["first"].flags.c_contiguous
    assert np.shares_memory(loads["first"], loading.link_loads)
    assert np.shares_memory(matrices["first"], od.demand)
    views = [inputs.masks, loading.link_loads, od.demand, loads["first"], matrices["first"]]
    for view in views:
        with pytest.raises(ValueError):
            view.flat[0] = 7
        with pytest.raises(ValueError):
            view.flags.writeable = True
    snapshots = [view.copy() for view in views]
    del inputs, loading, od, loads, matrices
    gc.collect()
    for view, expected in zip(views, snapshots, strict=True):
        np.testing.assert_array_equal(view, expected)


@pytest.mark.parametrize("selections", [[], None, {"": []}, {3: []}, {"a": [-1]}, {"a": [4]},
                                          {"a": [1.5]}, {"a": [True]}, {"a": [np.bool_(False)]}])
def test_invalid_selection_inputs(selections):
    with pytest.raises((TypeError, ValueError)):
        SelectLinkContext(4, selections)


@pytest.mark.parametrize("factory,args", [
    (SelectLinkContext, (0, {})),
    (SelectLinkLoadingOutputs, (0, 0, ())),
    (SelectLinkODOutputs, (0, 0, 0, ())),
    (SelectLinkOutputs, (0, 0, 0, ())),
])
def test_select_link_fixed_owners_cannot_be_reinitialized(factory, args):
    owner = factory(*args)
    with pytest.raises(RuntimeError):
        owner.__init__(*args)


@pytest.mark.parametrize("names", ["set", [""], [None], [1], ["same", "same"]])
def test_invalid_output_names(names):
    for factory, args in ((SelectLinkLoadingOutputs, (4, 1, names)),
                          (SelectLinkODOutputs, (1, 4, 1, names)),
                          (SelectLinkOutputs, (4, 4, 1, names))):
        with pytest.raises((TypeError, ValueError)):
            factory(*args)


@pytest.mark.parametrize("factory,args", [
    (SelectLinkContext, (-1, {})),
    (SelectLinkLoadingOutputs, (-1, 1, ())), (SelectLinkLoadingOutputs, (1, -1, ())),
    (SelectLinkODOutputs, (-1, 1, 1, ())), (SelectLinkODOutputs, (1, -1, 1, ())),
    (SelectLinkODOutputs, (1, 1, -1, ())), (SelectLinkOutputs, (-1, 1, 1, ())),
])
def test_invalid_output_dimensions(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


def test_validation_precedes_all_scratch_and_output_writes():
    context = history_context()
    results = search(context, 0)
    inputs = SelectLinkContext(4, {"all": range(4), "some": [2]})
    output = inputs.make_outputs(4, 2)
    flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 2)
    query = LoadingQuery(np.ones((4, 2)))
    good = dict(results=results, query=query, context=inputs, selection_workspace=flags,
                loading_workspace=loading, loading_output=output.loading, od_output=output.od)
    select_link_loading(**good)
    buffers = [flags.selected_paths, loading.state_loads, output.loading.link_loads, output.od.demand]
    snapshots = [buffer.copy() for buffer in buffers]
    bad_arguments = [
        {"context": SelectLinkContext(5, {"all": range(4), "some": [2]})},
        {"query": LoadingQuery(np.ones((5, 2)))},
        {"selection_workspace": SelectLinkWorkspace(context.state_count + 1)},
        {"loading_workspace": None},
        {"loading_workspace": LoadingWorkspace(context.state_count + 1, 2)},
        {"loading_workspace": LoadingWorkspace(context.state_count, 1)},
        {"loading_output": SelectLinkLoadingOutputs(5, 2, inputs.set_names)},
        {"loading_output": SelectLinkLoadingOutputs(4, 1, inputs.set_names)},
        {"loading_output": SelectLinkLoadingOutputs(4, 2, inputs.set_names[::-1])},
        {"od_output": SelectLinkODOutputs(1, 3, 2, inputs.set_names)},
        {"od_output": SelectLinkODOutputs(1, 4, 1, inputs.set_names)},
        {"od_output": SelectLinkODOutputs(1, 4, 2, inputs.set_names[::-1])},
        {"origin_row": -1}, {"origin_row": 1}, {"origin_row": 1.5},
    ]
    for change in bad_arguments:
        with pytest.raises((ValueError, TypeError)):
            select_link_loading(**(good | change))
        for buffer, expected in zip(buffers, snapshots, strict=True):
            np.testing.assert_array_equal(buffer, expected)


@pytest.mark.parametrize("turn", [False, True])
def test_workers_share_od_rows_and_reduce_only_link_loads(turn):
    context = history_context(turn=turn)
    selections = {"first": [0, 3], "last": [2]}
    inputs = SelectLinkContext(4, selections)
    output = inputs.make_outputs(4, 2, origin_count=4)
    demand = np.arange(32, dtype=np.float64).reshape(4, 4, 2)

    def run(origins):
        results = allocate_results(context)
        flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 2)
        worker = SelectLinkLoadingOutputs(4, 2, inputs.set_names)
        for origin in origins:
            search(context, origin, results=results)
            select_link_loading(results, LoadingQuery(demand[origin]), inputs, flags, loading,
                                worker, output.od, origin_row=origin)
        return worker

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = list(pool.map(run, ([0, 2], [1, 3])))
    expected = path_walk_outputs(context, demand, selected_links=list(selections.values()))
    snapshots = [worker.link_loads.copy() for worker in workers]
    od_snapshot = output.od.demand.copy()
    retained = output.loading.link_loads
    for _ in range(3):
        assert reduce_select_link_loading_outputs(iter(workers), output.loading) is output.loading
        np.testing.assert_array_equal(retained, expected[3])
        np.testing.assert_array_equal(output.od.demand, expected[4])
        np.testing.assert_array_equal(output.od.demand, od_snapshot)
        for worker, snapshot in zip(workers, snapshots, strict=True):
            np.testing.assert_array_equal(worker.link_loads, snapshot)
    for invalid in ([output.loading], [workers[0], None], [object()],
                    [SelectLinkLoadingOutputs(3, 2, inputs.set_names)],
                    [SelectLinkLoadingOutputs(4, 3, inputs.set_names)],
                    [SelectLinkLoadingOutputs(4, 2, inputs.set_names[::-1])]):
        with pytest.raises((ValueError, TypeError)):
            reduce_select_link_loading_outputs(invalid, output.loading)
        np.testing.assert_array_equal(retained, expected[3])
    reduce_select_link_loading_outputs([], output.loading)
    assert not np.any(retained)
    assert retained.ctypes.data == output.loading.link_loads.ctypes.data


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("link_loads, od", [(True, True), (True, False), (False, True), (False, False)])
def test_assignment_optional_components_match_standalone_and_outlive_driver(turn, link_loads, od):
    context = history_context(turn=turn)
    inputs = SelectLinkContext(4, {"first": [0, 3], "last": [2]})
    demand = np.ones((4, 4, 2))
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=3, selected_links=inputs,
                           select_link_loads=link_loads, select_link_od=od, origins=[0, 2])
    assigned = prepared.make_outputs()
    # Start with all rows populated; the subset run must clear the skipped ones.
    full = PreparedAoN(context, demand, costs=context.costs, selected_links=inputs,
                       select_link_loads=link_loads, select_link_od=od)
    full.run(assigned)
    standalone = inputs.make_outputs(4, 2, origin_count=4, link_loads=link_loads, od=od)
    flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 2)
    # Both paths replace an iteration, rather than accumulating previous runs.
    for _ in range(3):
        standalone.reset()
        prepared.run(assigned)
        for origin in (0, 2):
            select_link_loading(search(context, origin), LoadingQuery(demand[origin]), inputs, flags,
                                loading if link_loads else None, standalone.loading, standalone.od,
                                origin_row=origin)
        if link_loads:
            np.testing.assert_array_equal(assigned.select_link_loads, standalone.loading.link_loads)
        else:
            assert assigned.select_link_loads is None
        if od:
            np.testing.assert_array_equal(assigned.select_link_od, standalone.od.demand)
            assert not np.any(assigned.select_link_od[[1, 3]])
        else:
            assert assigned.select_link_od is None
    component = assigned.select_link
    del prepared, full, assigned
    component.reset()
    select_link_loading(search(context, 0), LoadingQuery(demand[0]), inputs, flags,
                        loading if link_loads else None, component.loading, component.od)
    if link_loads:
        np.testing.assert_array_equal(component.loading.link_loads,
                                      walk_selected_paths(search(context, 0), demand[0], {"first": [0, 3], "last": [2]})[0])


def test_assignment_checks_selection_configuration_before_resetting_output():
    context = history_context()
    inputs = SelectLinkContext(4, {"first": [0, 3], "last": [2]})
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, selected_links=inputs)
    out = prepared.run(prepared.make_outputs())
    snapshots = out.link_loads.copy(), out.select_link_loads.copy(), out.select_link_od.copy()
    incompatible = PreparedAoN(context, demand, costs=context.costs, selected_links=inputs, select_link_loads=False)
    with pytest.raises(ValueError, match="components"):
        incompatible.run(out)
    for values, expected in zip((out.link_loads, out.select_link_loads, out.select_link_od), snapshots, strict=True):
        np.testing.assert_array_equal(values, expected)
    reversed_names = AoNOutputs(4, 4, 1, select_link_names=inputs.set_names[::-1])
    with pytest.raises(ValueError, match="names"):
        prepared.run(reversed_names)
    with pytest.raises(ValueError, match="link_count"):
        PreparedAoN(context, demand, costs=context.costs, selected_links=SelectLinkContext(5, {}))
    with pytest.raises(TypeError):
        PreparedAoN(context, demand, costs=context.costs, selected_links={"first": [0]})


def test_no_outputs_or_no_sets_leaves_scratch_unchanged():
    context = history_context()
    results = search(context, 0)
    inputs = SelectLinkContext(4, {"all": range(4)})
    output = inputs.make_outputs(4, 1)
    flags, loading = SelectLinkWorkspace(context.state_count), LoadingWorkspace(context.state_count, 1)
    query = LoadingQuery(np.ones((4, 1)))
    select_link_loading(results, query, inputs, flags, loading, output.loading, output.od)
    flag_snapshot, load_snapshot = flags.selected_paths.copy(), loading.state_loads.copy()
    empty_results = allocate_results(context)
    select_link_loading(empty_results, query, inputs, flags, loading)
    empty_inputs = SelectLinkContext(4, {})
    empty_output = empty_inputs.make_outputs(4, 1)
    select_link_loading(empty_results, query, empty_inputs, flags, loading,
                        empty_output.loading, empty_output.od)
    np.testing.assert_array_equal(flags.selected_paths, flag_snapshot)
    np.testing.assert_array_equal(loading.state_loads, load_snapshot)


def test_masks_are_retained_by_driver_but_not_its_outputs():
    context = history_context()
    inputs = TrackedSelectLinkContext(4, {"all": range(4)})
    input_ref = weakref.ref(inputs)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs, selected_links=inputs)
    del inputs
    gc.collect()
    assert input_ref() is not None
    out = prepared.run(prepared.make_outputs())
    del prepared
    gc.collect()
    assert input_ref() is None
    assert np.any(out.select_link_loads)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("link_loads, od", [(True, True), (True, False), (False, True)])
def test_no_active_origins_clears_reused_selected_outputs(turn, link_loads, od):
    context = history_context(turn=turn)
    inputs = SelectLinkContext(4, {"all": range(4)})
    active = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs, selected_links=inputs,
                        select_link_loads=link_loads, select_link_od=od, cores=3)
    out = active.run(active.make_outputs())
    empty = PreparedAoN(context, np.zeros((4, 4, 1)), costs=context.costs, selected_links=inputs,
                       select_link_loads=link_loads, select_link_od=od, cores=3)
    empty.run(out)
    if link_loads:
        assert not np.any(out.select_link_loads)
    if od:
        assert not np.any(out.select_link_od)
