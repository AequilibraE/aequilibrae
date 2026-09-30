"""Assignment composes independent owners and reuses them as routing costs change."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.context import SkimmingContext, SelectLinkContext
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.outputs import AoNOutputs, LoadingOutputs, SelectLinkLoadingOutputs
from aequilibrae.paths.cython.network_loading import network_loading
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.workspaces import AoNWorkspace, LoadingWorkspace
from .routing_helpers import history_context, make_context, path_walk_outputs, search


def output_arrays(output):
    """Retain every allocated component array, without aggregate aliases."""
    arrays = [output.loading.link_loads]
    if output.skimming is not None:
        arrays.append(output.skimming.skims)
    if output.select_link is not None:
        if output.select_link.loading is not None:
            arrays.append(output.select_link.loading.link_loads)
        if output.select_link.od is not None:
            arrays.append(output.select_link.od.demand)
    return arrays


def assert_walk_outputs(output, expected):
    np.testing.assert_allclose(output.loading.link_loads, expected[0])
    np.testing.assert_allclose(output.skimming.skims, expected[1])
    assert output.turn_cost_total == expected[2]
    np.testing.assert_allclose(output.select_link.loading.link_loads, expected[3])
    np.testing.assert_allclose(output.select_link.od.demand, expected[4])


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
def test_selected_full_paths_and_output_rotation(turn, cores):
    context = history_context(turn=turn)
    demand = np.ones((4, 4, 2))
    selections = {"screenline": [0, 3, 3], "other": [2], "empty": []}
    fields = [np.ones(context.link_count)]
    skims = SkimmingContext(context.link_count, link_fields={"time": fields[0]})
    prepared = PreparedAoN(
        context, demand, cores=cores, selected_links=SelectLinkContext(context.link_count, selections), skimming=skims
    )
    previous, current = prepared.make_outputs(), prepared.make_outputs()
    retained = output_arrays(previous)
    for iteration in range(4):
        context.update_costs(np.array([1.0, 1.0 + iteration * 5, 1.0, 1.0]))
        expected = path_walk_outputs(context, demand, fields, selected_links=list(selections.values()))
        saved = [values.copy() for values in output_arrays(previous)]
        saved_total = previous.turn_cost_total
        assert prepared.run(current) is current
        for values, snapshot in zip(output_arrays(previous), saved, strict=True):
            np.testing.assert_array_equal(values, snapshot)
        assert previous.turn_cost_total == saved_total
        assert_walk_outputs(current, expected)
        previous, current = current, previous
    for values, component in zip(retained, output_arrays(previous), strict=True):
        assert np.shares_memory(values, component)


@pytest.mark.parametrize("cores", [1, 3])
def test_worker_turn_totals_accumulate_across_origins_and_reset_between_runs(cores):
    # Each origin in this directed cycle pays one turn to reach its second
    # destination. A single worker must retain all three origin contributions.
    context = make_context(
        [0, 1, 2, 3],
        [1, 2, 0],
        [1, 1, 1],
        turns={(0, 1): 2.0, (1, 2): 3.0, (2, 0): 4.0},
        turn=True,
    )
    demand = np.ones((3, 3, 2))
    prepared = PreparedAoN(context, demand, cores=cores)
    output = prepared.make_outputs()
    expected_total = 2 * (2.0 + 3.0 + 4.0)

    for _ in range(3):
        context.update_costs(np.ones(3))
        prepared.run(output)
        assert output.turn_cost_total == expected_total

        # This next run contributes nothing. Reusing the same workers must not
        # carry the previous iteration's scalar totals into its reduction.
        context.update_costs(np.full(3, np.inf))
        prepared.run(output)
        assert output.turn_cost_total == 0


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("seed", [17, 29, 42])
def test_random_selected_loading_against_path_walks(turn, seed):
    rng = np.random.default_rng(seed)
    nodes = 8
    edges = [
        (a, b, float(rng.integers(0, 5)))
        for a in range(nodes)
        for b in range(nodes)
        for _ in range(2)
        if rng.random() < 0.1
    ]
    turns = {}
    if turn:
        for incoming, (_, via, _) in enumerate(edges):
            for outgoing, (tail, _, _) in enumerate(edges):
                if via == tail and rng.random() < 0.25:
                    turns[incoming, outgoing] = np.inf if rng.random() < 0.2 else 5.0
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=nodes))]
    context = make_context(fs, [b for _, b, _ in edges], [c for _, _, c in edges], turns if turn else None, turn=turn)
    demand = rng.integers(-2, 5, size=(nodes, nodes, 2)).astype(np.float64)
    selected = {"first": list(range(0, context.link_count, 3)), "second": list(range(1, context.link_count, 2))}
    fields = [rng.random(context.link_count)]
    skims = SkimmingContext(context.link_count, link_fields={"distance": fields[0]})
    prepared = PreparedAoN(
        context, demand, cores=3, selected_links=SelectLinkContext(context.link_count, selected), skimming=skims
    )
    out = prepared.run(prepared.make_outputs())
    assert_walk_outputs(out, path_walk_outputs(context, demand, fields, selected_links=list(selected.values())))


@pytest.mark.parametrize("turn", [False, True])
def test_caller_creates_independent_cost_bindings_and_outputs_retain_no_inputs(turn):
    context = history_context(turn=turn)
    demand = np.ones((4, 4, 1))
    costs = np.array([1.0, 20.0, 1.0, 1.0] if turn else [20.0, 1.0, 1.0, 1.0])
    costs_ref = weakref.ref(costs)
    routing = context.with_costs(costs)
    a = PreparedAoN(context, demand)
    b = PreparedAoN(routing, demand)
    first, second = a.run(a.make_outputs()), b.run(b.make_outputs())
    assert not np.array_equal(first.loading.link_loads, second.loading.link_loads)
    assert np.shares_memory(routing.costs, costs)
    del costs, routing
    gc.collect()
    assert costs_ref() is not None  # The driver retains the caller's context.
    b.run(second)
    del b
    gc.collect()
    assert costs_ref() is None
    view = second.loading.link_loads
    snapshot = view.copy()
    del second
    gc.collect()
    np.testing.assert_array_equal(view, snapshot)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("rebind", [False, True])
def test_cost_updates_refresh_routing_without_repreparing_workers(turn, rebind, monkeypatch):
    costs = np.ones(4)
    context = history_context(turn=turn).with_costs(costs)
    demand = np.ones((4, 4, 1))
    skims = SkimmingContext(4, cost_name="objective")
    prepared = PreparedAoN(
        context, demand, skimming=skims, cores=3, selected_links=SelectLinkContext(4, {"all": range(4)})
    )
    out = prepared.run(prepared.make_outputs())
    initial_loads = out.loading.link_loads.copy()
    output_views = output_arrays(out)
    masks, origins, counts = prepared.destination_masks, prepared.origins, prepared.destination_counts

    # Inspect retained owners through GC rather than adding a worker-buffer API
    # just for tests. Every search, workspace and accumulator must be reused.
    (worker_list,) = [value for value in gc.get_referents(prepared) if isinstance(value, list)]
    owners = [
        value
        for worker in worker_list
        for value in gc.get_referents(worker)
        if isinstance(value, (SearchResults, AoNWorkspace, LoadingOutputs, SelectLinkLoadingOutputs))
    ]
    assert len(owners) == 3 * 4

    def buffers(owner):
        if isinstance(owner, SearchResults):
            return [
                owner.predecessors,
                owner.connectors,
                owner.settlement_order,
                owner.terminal_states,
                owner.distances,
                owner.turn_costs,
            ]
        if isinstance(owner, AoNWorkspace):
            assert owner.skimming is None  # Objective projection needs no state sums.
            return [owner.loading.state_loads, owner.select_link.selected_paths]
        return [owner.link_loads]

    addresses = [[values.ctypes.data for values in buffers(owner)] for owner in owners]

    def unexpected(*args, **kwargs):
        raise AssertionError("setup should not run inside an iteration")

    for name in ("borrow_input", "choose_origins", "make_destination_masks"):
        monkeypatch.setattr(f"aequilibrae.paths.cython.aon_context.{name}", unexpected)

    for iteration in range(3):
        values = [1.0, 20.0 + iteration, 1.0, 1.0] if turn else [20.0 + iteration, 1.0, 1.0, 1.0]
        if rebind:
            old = weakref.ref(costs)
            costs = np.array(values)
            context.update_costs(costs)
            gc.collect()
            assert old() is None  # A cached routing view must not keep this pointer.
        else:
            costs[:] = values
        prepared.run(out)
        np.testing.assert_array_equal(out.loading.link_loads, path_walk_outputs(context, demand)[0])
        assert not np.array_equal(out.loading.link_loads, initial_loads)
        for origin in range(4):
            results = search(context, origin)
            expected = [results.path_cost_to(node) for node in range(4)]
            np.testing.assert_array_equal(out.skimming.matrices["objective"][origin], expected)
        for owner, expected_addresses in zip(owners, addresses, strict=True):
            assert [values.ctypes.data for values in buffers(owner)] == expected_addresses
        for previous, current in zip(output_views, output_arrays(out), strict=True):
            assert previous.ctypes.data == current.ctypes.data
        assert masks.ctypes.data == prepared.destination_masks.ctypes.data
        assert origins.ctypes.data == prepared.origins.ctypes.data
        assert counts.ctypes.data == prepared.destination_counts.ctypes.data

    saved = [values.copy() for values in output_arrays(out)]
    with pytest.raises(ValueError):
        context.update_costs(np.full(4, -1.0))
    prepared.run(out)
    for values, expected in zip(output_arrays(out), saved, strict=True):
        np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("turn", [False, True])
def test_drivers_can_share_a_context_and_observe_rebinding_between_runs(turn):
    context = history_context(turn=turn)
    demand = np.ones((4, 4, 1))
    drivers = [PreparedAoN(context, demand, cores=2) for _ in range(2)]
    outputs = [driver.make_outputs() for driver in drivers]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for costs in (np.ones(4), np.array([20.0, 15.0, 1.0, 1.0])):
            context.update_costs(costs)
            expected = path_walk_outputs(context, demand)[0]
            futures = [pool.submit(driver.run, output) for driver, output in zip(drivers, outputs, strict=True)]
            for future, output in zip(futures, outputs, strict=True):
                assert future.result() is output
                np.testing.assert_array_equal(output.loading.link_loads, expected)


@pytest.mark.parametrize("turn", [False, True])
def test_centroid_blocking_is_configured_on_the_context(turn):
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 10, 1], turn=turn, blocked_centroid_count=3)
    heads = context.heads.copy()
    demand = np.zeros((3, 3, 1))
    demand[0, 2, 0] = 1
    prepared = PreparedAoN(context, demand, cores=3)
    out = prepared.run(prepared.make_outputs())
    np.testing.assert_array_equal(out.loading.link_loads[:, 0], [0, 1, 0])
    np.testing.assert_array_equal(context.heads, heads)
    assert context.blocked_centroid_count == 3


@pytest.mark.parametrize("turn", [False, True])
def test_no_active_origins_resets_outputs(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    prepared = PreparedAoN(context, np.zeros((2, 2, 1)), selected_links=SelectLinkContext(0, {"empty": []}))
    out = prepared.run(prepared.make_outputs())
    assert out.turn_cost_total == 0
    assert np.all(out.select_link.od.demand == 0)
    assert out.select_link.loading.link_loads.shape == (1, 0, 1)


@pytest.mark.parametrize("skims", [False, True])
@pytest.mark.parametrize("selected_loads, selected_od", [(False, False), (False, True), (True, False), (True, True)])
def test_output_group_allocates_resets_and_releases_independent_components(skims, selected_loads, selected_od):
    context = history_context(0.5)
    prepared = PreparedAoN(
        context,
        np.ones((4, 4, 1)),
        skimming=SkimmingContext(4, cost_name="cost") if skims else None,
        selected_links=SelectLinkContext(4, {"all": range(4)}),
        select_link_loads=selected_loads,
        select_link_od=selected_od,
    )
    out = prepared.run(prepared.make_outputs())
    assert isinstance(out, AoNOutputs)
    assert (out.skimming is not None) == skims
    assert (out.select_link is not None) == (selected_loads or selected_od)
    if out.select_link is not None:
        assert (out.select_link.loading is not None) == selected_loads
        assert (out.select_link.od is not None) == selected_od
    assert out.turn_cost_total > 0
    retained = output_arrays(out)

    out.reset()
    assert out.turn_cost_total == 0
    assert not np.any(out.loading.link_loads)
    if skims:
        assert np.all(np.isinf(out.skimming.skims))
    if selected_loads:
        assert not np.any(out.select_link.loading.link_loads)
    if selected_od:
        assert not np.any(out.select_link.od.demand)

    for previous, current in zip(retained, output_arrays(out), strict=True):
        assert previous.ctypes.data == current.ctypes.data
        assert not current.flags.writeable

    loading = out.loading
    skimming = out.skimming
    selection = out.select_link
    for name in (
        "shape",
        "links",
        "zones",
        "classes",
        "fields",
        "skim_names",
        "select_link_names",
        "link_loads",
        "skims",
        "select_link_loads",
        "select_link_od",
        "total_turn_penalty",
    ):
        assert not hasattr(out, name)

    del prepared, out
    loading.reset()
    network_loading(
        search(context, 0), LoadingQuery(np.ones((4, 1))), LoadingWorkspace(context.state_count, 1), loading
    )
    assert np.any(retained[0])
    if skimming is not None:
        skimming.reset()
    if selection is not None:
        selection.reset()


@pytest.mark.parametrize("dimensions", [(0, 0, 0), (0, 2, 1), (2, 0, 1), (2, 3, 0)])
def test_output_allocator_supports_empty_axes_and_fixed_components(dimensions):
    out = AoNOutputs(*dimensions, skim_names=("cost",), select_link_names=("set",))
    links, zones, classes = dimensions
    assert out.loading.link_loads.shape == (links, classes)
    assert out.skimming.skims.shape == (zones, 1, zones)
    assert out.select_link.loading.link_loads.shape == (1, links, classes)
    assert out.select_link.od.demand.shape == (zones, 1, zones, classes)
    out.reset()
    with pytest.raises(RuntimeError):
        out.__init__(*dimensions)
    with pytest.raises(AttributeError):
        out.loading = LoadingOutputs(links, classes)


@pytest.mark.parametrize(
    "change",
    [
        "links",
        "classes",
        "zones",
        "skim_absent",
        "skim_names",
        "selection_absent",
        "selection_names",
        "loading_only",
        "od_only",
    ],
)
def test_driver_checks_actual_components_before_any_output_writes(change):
    context = history_context(0.5)
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(
        context,
        demand,
        skimming=SkimmingContext(4, cost_name="cost"),
        selected_links=SelectLinkContext(4, {"set": range(4)}),
    )
    other_context = context
    if change == "links":
        other_context = make_context([0, 3, 4, 5, 5], [0, 1, 2, 3, 1], [1] * 5)
    zones = 3 if change == "zones" else 4
    classes = 2 if change == "classes" else 1

    skimming = None
    if change != "skim_absent":
        cost_name = "different" if change == "skim_names" else "cost"
        skimming = SkimmingContext(other_context.link_count, cost_name=cost_name)

    selections = None
    if change != "selection_absent":
        set_name = "different" if change == "selection_names" else "set"
        selections = SelectLinkContext(other_context.link_count, {set_name: range(4)})

    other = PreparedAoN(
        other_context,
        np.ones((zones, zones, classes)),
        skimming=skimming,
        selected_links=selections,
        select_link_loads=change != "od_only",
        select_link_od=change != "loading_only",
    )
    out = other.run(other.make_outputs())
    snapshots = [values.copy() for values in output_arrays(out)]
    total = out.turn_cost_total

    with pytest.raises(ValueError):
        prepared.run(out)
    for values, expected in zip(output_arrays(out), snapshots, strict=True):
        np.testing.assert_array_equal(values, expected)
    assert out.turn_cost_total == total

    # A rejected run also leaves the driver usable with the right components.
    prepared.run(prepared.make_outputs())


def test_driver_has_no_routing_configuration_proxies():
    context = history_context()
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand)
    for name in ("costs", "update_costs", "shape"):
        assert not hasattr(prepared, name)
    for options in ({"costs": context.costs}, {"block_centroids": True}):
        with pytest.raises(TypeError):
            PreparedAoN(context, demand, **options)
    with pytest.raises(RuntimeError):
        prepared.__init__(context, demand)
    for bad in (demand.astype(np.float32), np.ones((4, 8, 1))[:, ::2], demand.tolist()):
        with pytest.raises((ValueError, TypeError)):
            PreparedAoN(context, bad)
    with pytest.raises(ValueError, match="contiguous"):
        SkimmingContext(context.link_count, link_fields={"distance": np.ones(8)[::2]})
