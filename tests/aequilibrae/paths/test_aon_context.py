"""End-to-end adapter tests using real Graph, matrix and assignment buffers."""

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.AoN import aon_parallel, aon_parallel_context
from aequilibrae.paths.cython.aon_context import AoNOutputs, PreparedAoN
from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext
from aequilibrae.utils.cython.array_allocations import array as allocate_array

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.cython.aon_graph import prepare_aon
from aequilibrae.paths.multi_threaded_aon import MultiThreadedAoN
from aequilibrae.paths.results import AssignmentResults


def history_graph(penalty=None, *, skimming=True, blocked=False):
    # All four nodes are centroids, so the turn-history example is uncompressed.
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4],
            "a_node": [10, 10, 20, 30],
            "b_node": [20, 30, 40, 20],
            "direction": [1] * 4,
            "cost": [1.0] * 4,
            "distance": [10.0, 20.0, 30.0, 40.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(blocked)
    graph.set_graph("cost")
    if penalty is not None:
        graph.set_turn_restrictions(
            pd.DataFrame(
                {
                    "from_node": [10],
                    "via_node": [20],
                    "to_node": [40],
                    "penalty": [penalty],
                }
            )
        )
    graph.set_skimming(["cost", "distance"] if skimming else [])
    return graph


def make_matrix(graph, classes=2):
    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=graph.num_zones, matrix_names=[f"class_{i}" for i in range(classes)])
    matrix.index[:] = graph.centroids
    matrix.computational_view()
    matrix.matrix_view = matrix.matrix_view.reshape(graph.num_zones, graph.num_zones, classes)
    matrix.matrix_view.fill(0)
    return matrix


def buffers(graph, matrix, cores, selected_links=None):
    result = AssignmentResults()
    result.set_cores(cores)
    if selected_links is not None:
        result._selected_links = selected_links
    result.prepare(graph, matrix)
    aux = MultiThreadedAoN()
    aux.prepare(graph, result)
    return result, aux


@pytest.mark.parametrize("penalty", [None, 0.5, 10.0, np.inf])
@pytest.mark.parametrize("cores", [1, 3])
@pytest.mark.parametrize("skimming", [False, True])
def test_search_skim_load_and_reduce(penalty, cores, skimming):
    graph = history_graph(penalty, skimming=skimming)
    matrix = make_matrix(graph)
    matrix.matrix_view[0] = [[999, 999], [2, 20], [3, 30], [5, 50]]
    # Another origin must accumulate into the same link after the scratch reset.
    matrix.matrix_view[1, 3] = [7, 70]
    result, aux = buffers(graph, matrix, cores)
    heads = graph.compact_graph.b_node.to_numpy().copy()
    costs = graph.compact_cost.copy()
    demand_snapshot = matrix.matrix_view.copy()
    skim_output = result.skims.matrix_view
    loads_output = aux.temp_link_loads
    report = aon_parallel_context(matrix, graph, result, aux, cores, bridge=None)
    assert report == (["Centroid 40 is not connected"] if skimming else [])
    assert aux.temp_link_loads is loads_output
    assert result.skims.matrix_view is skim_output

    detour = penalty is not None and penalty >= 10
    expected = np.array([2, 8, 12, 5] if detour else [7, 3, 12, 0])[:, None] * [1, 10]
    np.testing.assert_array_equal(aux.temp_link_loads.sum(axis=0)[:4], expected)
    np.testing.assert_array_equal(aux.temp_link_loads[:, 4:], 0)  # Sentinel remains zero.
    assert aux.turn_penalty_accumulator.sum() == (27.5 if penalty == 0.5 else 0.0)
    if skimming:
        expected_skims = [[0, 0], [1, 10], [1, 20], [3, 90] if detour else [2 + (penalty or 0), 40]]
        np.testing.assert_array_equal(result.skims.matrix_view[0], expected_skims)
        assert np.isinf(result.skims.matrix_view[1, 0]).all()
        # Disconnected origins are reported, and their existing rows are not overwritten.
        np.testing.assert_array_equal(result.skims.matrix_view[3], 0)
    np.testing.assert_array_equal(matrix.matrix_view, demand_snapshot)
    np.testing.assert_array_equal(graph.compact_graph.b_node, heads)
    np.testing.assert_array_equal(graph.compact_cost, costs)

    # Caller-owned accumulators are not silently reset. Then explicitly start
    # another iteration, changing costs to check that snapshots are refreshed.
    aon_parallel_context(matrix, graph, result, aux, cores)
    np.testing.assert_array_equal(aux.temp_link_loads.sum(axis=0)[:4], expected * 2)
    aux.temp_link_loads.fill(0)
    aux.turn_penalty_accumulator.fill(0)
    graph.compact_cost *= 2
    aon_parallel_context(matrix, graph, result, aux, cores)
    np.testing.assert_array_equal(aux.temp_link_loads.sum(axis=0)[:4], expected)


@pytest.mark.parametrize("selected_fields", [[], ["distance"], ["cost", "distance"]])
def test_selected_penalty_skim_fields(selected_fields):
    graph = history_graph(0.5)
    graph.turn_skim_fields = selected_fields
    matrix = make_matrix(graph, classes=1)
    matrix.matrix_view[0, 3, 0] = 10
    result, aux = buffers(graph, matrix, 2)
    aon_parallel_context(matrix, graph, result, aux, 2)
    expected = [2.0, 40.0]
    for i, name in enumerate(graph.skim_fields):
        if name in (selected_fields or ["cost"]):
            expected[i] += 0.5
    np.testing.assert_array_equal(result.skims.matrix_view[0, 3], expected)
    assert aux.turn_penalty_accumulator.sum() == 5


@pytest.mark.parametrize("blocked", [False, True])
def test_adapter_matches_legacy_for_connector_bans(blocked):
    graph = history_graph(blocked=blocked)
    assert graph.has_turn_restrictions == blocked
    matrix = make_matrix(graph)
    matrix.matrix_view[:] = np.random.default_rng(41).random(matrix.matrix_view.shape)
    old, old_aux = buffers(graph, matrix, 2)
    new, new_aux = buffers(graph, matrix, 2)
    old_report = aon_parallel(matrix, graph, old, old_aux, 2)
    new_report = aon_parallel_context(matrix, graph, new, new_aux, 2)
    assert new_report == old_report
    np.testing.assert_allclose(new_aux.temp_link_loads.sum(axis=0), old_aux.temp_link_loads.sum(axis=0))
    np.testing.assert_allclose(new.skims.matrix_view, old.skims.matrix_view)


@pytest.mark.parametrize("select_links", [False, True])
def test_compressed_network_through_existing_assignment_caller(monkeypatch, select_links):
    # Degree-two intermediate nodes compress into links between centroids. Full
    # link mapping, skim expansion and reduction are the real production caller's.
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4],
            "a_node": [10, 11, 20, 21],
            "b_node": [11, 20, 21, 30],
            "direction": [0] * 4,
            "cost": [1.0, 2.0, 3.0, 4.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 30]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("cost")
    graph.set_skimming(["cost"])
    assert graph.compact_num_links < graph.num_links
    matrix = make_matrix(graph, classes=1)
    matrix.matrix_view[:] = np.arange(9.0).reshape(3, 3, 1)
    selected = None
    if select_links:
        selected = {"screenline": graph.graph.loc[graph.graph.link_id == 1, "__compressed_id__"].to_numpy()}
    old, _ = buffers(graph, matrix, 2, selected)
    baseline = allOrNothing("baseline", matrix, graph, old)
    baseline.execute()

    monkeypatch.setattr("aequilibrae.paths.all_or_nothing.aon_parallel", aon_parallel_context)
    new, _ = buffers(graph, matrix, 2, selected)
    assignment = allOrNothing("context", matrix, graph, new)
    assignment.execute()
    np.testing.assert_allclose(new.link_loads, old.link_loads)
    np.testing.assert_allclose(new.compact_link_loads, old.compact_link_loads)
    np.testing.assert_allclose(new.skims.matrix_view, old.skims.matrix_view)
    assert assignment.report == baseline.report
    assert np.any(new.link_loads > 0)
    if select_links:
        np.testing.assert_allclose(assignment.aux_res.temp_sl_link_loading.sum(axis=0),
                                   baseline.aux_res.temp_sl_link_loading.sum(axis=0))
        np.testing.assert_allclose(assignment.aux_res.temp_sl_od_matrix.sum(axis=0),
                                   baseline.aux_res.temp_sl_od_matrix.sum(axis=0))
        assert np.any(assignment.aux_res.temp_sl_link_loading > 0)


@pytest.mark.parametrize("penalty", [0.5, 10.0])
def test_turn_network_through_existing_assignment_caller(monkeypatch, penalty):
    graph = history_graph(penalty)
    matrix = make_matrix(graph)
    matrix.matrix_view[0] = [[0, 0], [2, 20], [3, 30], [5, 50]]
    result, _ = buffers(graph, matrix, 2)
    monkeypatch.setattr("aequilibrae.paths.all_or_nothing.aon_parallel", aon_parallel_context)
    assignment = allOrNothing("context", matrix, graph, result)
    assignment.execute()
    expected = np.array([7, 3, 5, 0] if penalty == 0.5 else [2, 8, 5, 5])[:, None] * [1, 10]
    np.testing.assert_array_equal(result.link_loads, expected)
    assert result.total_turn_penalty == (27.5 if penalty == 0.5 else 0.0)
    np.testing.assert_array_equal(result.skims.matrix_view[0, 3], [2.5, 40] if penalty == 0.5 else [3, 90])


def test_node_mode_centroid_blocking_keeps_context_heads_immutable():
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [1, 2, 3],
            "a_node": [10, 11, 12],
            "b_node": [11, 12, 20],
            "direction": [0] * 3,
            "cost": [1.0, 2.0, 3.0],
        }
    )
    graph.prepare_graph(np.array([10, 20]), remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming(["cost"])
    assert graph.block_centroid_flows and not graph.has_turn_restrictions
    matrix = make_matrix(graph, classes=1)
    matrix.matrix_view[:, :, 0] = [[0, 10], [20, 0]]
    old, old_aux = buffers(graph, matrix, 2)
    new, new_aux = buffers(graph, matrix, 2)
    heads = graph.compact_graph.b_node.to_numpy().copy()
    aon_parallel(matrix, graph, old, old_aux, 2)
    for _ in range(2):
        new_aux.temp_link_loads.fill(0)
        aon_parallel_context(matrix, graph, new, new_aux, 2)
        np.testing.assert_allclose(new_aux.temp_link_loads.sum(axis=0), old_aux.temp_link_loads.sum(axis=0))
        np.testing.assert_array_equal(new.skims.matrix_view[:, :, 0], [[0, 6], [6, 0]])
        np.testing.assert_array_equal(graph.compact_graph.b_node, heads)


def test_isolated_first_centroid_is_reported():
    graph = history_graph()
    with pytest.warns(UserWarning, match="centroids not present"):
        graph.prepare_graph(np.array([5, 10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming(["cost", "distance"])
    matrix = make_matrix(graph, classes=1)
    matrix.matrix_view[1, 4, 0] = 10
    result, aux = buffers(graph, matrix, 2)
    report = aon_parallel_context(matrix, graph, result, aux, 2)
    assert report == ["Centroid 5 is not connected", "Centroid 40 is not connected"]
    np.testing.assert_array_equal(aux.temp_link_loads.sum(axis=0)[:4, 0], [10, 0, 10, 0])
    np.testing.assert_array_equal(result.skims.matrix_view[1, 4], [2, 40])


def test_strided_matrix_subset_is_packed_once():
    graph = history_graph()
    matrix = make_matrix(graph, classes=3)
    matrix.computational_view(["class_1"])
    assert not matrix.matrix_view.flags.c_contiguous
    matrix.matrix_view[:] = np.arange(16.0).reshape(4, 4)
    result, aux = buffers(graph, matrix, 2)
    aon_parallel_context(matrix, graph, result, aux, 2)
    matrix.matrix_view = matrix.matrix_view.copy().reshape(4, 4, 1)
    reference, ref_aux = buffers(graph, matrix, 2)
    aon_parallel(matrix, graph, reference, ref_aux, 2)
    np.testing.assert_array_equal(aux.temp_link_loads.sum(axis=0), ref_aux.temp_link_loads.sum(axis=0))


def test_no_demand_and_no_skims_leaves_outputs_alone():
    graph = history_graph(skimming=False)
    matrix = make_matrix(graph)
    result, aux = buffers(graph, matrix, 2)
    aux.temp_link_loads.fill(7)
    assert aon_parallel_context(matrix, graph, result, aux, 2) == []
    np.testing.assert_array_equal(aux.temp_link_loads, 7)


@pytest.mark.parametrize(
    "option, value, message",
    [
        ("save_path_file", True, "path-file"),
        ("_heap", "pairing", "4ary"),
    ],
)
def test_unsupported_options_are_explicit(option, value, message):
    graph = history_graph()
    matrix = make_matrix(graph)
    result, aux = buffers(graph, matrix, 2)
    setattr(result, option, value)
    with pytest.raises(NotImplementedError, match=message):
        aon_parallel_context(matrix, graph, result, aux, 2)
    np.testing.assert_array_equal(aux.temp_link_loads, 0)


def test_validation_before_parallel_loop():
    graph = history_graph()
    matrix = make_matrix(graph)
    result, aux = buffers(graph, matrix, 2)
    for cores in (0, 3):
        with pytest.raises(ValueError):
            aon_parallel_context(matrix, graph, result, aux, cores)
    original = aux.temp_link_loads
    aux.temp_link_loads = original[:, ::2]
    with pytest.raises(ValueError, match="temp_link_loads"):
        aon_parallel_context(matrix, graph, result, aux, 2)
    aux.temp_link_loads = original
    good_skims = graph.compact_skims
    graph.compact_skims = graph.compact_skims[:1]
    with pytest.raises(ValueError, match="compact_skims"):
        aon_parallel_context(matrix, graph, result, aux, 2)
    graph.compact_skims = good_skims
    matrix.index[:] = matrix.index[::-1]
    with pytest.raises(ValueError, match="centroids"):
        aon_parallel_context(matrix, graph, result, aux, 2)
    result._graph_id = "wrong"
    with pytest.raises(ValueError, match="not prepared"):
        aon_parallel_context(matrix, graph, result, aux, 2)


def prepared_context(turn):
    args = ([0, 2, 3, 4, 4], [1, 2, 3, 1], [1.0] * 4)
    return TurnBasedContext(*args, [0, 1, 1, 1, 1], [2], [0.5]) if turn else NodeBasedContext(*args)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
@pytest.mark.parametrize("skimming", [False, True])
def test_prepared_iterations_snapshots_and_fixed_inputs(turn, cores, skimming, monkeypatch):
    context = prepared_context(turn)
    demand = np.zeros((4, 4, 2))
    demand[0, 1], demand[0, 3], demand[1, 3] = [2, 20], [5, 50], [7, 70]
    distance = np.array([10.0, 20.0, 30.0, 40.0])
    fields = [context.costs, distance] if skimming else []
    prepared = PreparedAoN(
        context, demand, costs=context.costs, cores=cores, skim_fields=fields,
        skim_penalties=[True, False] if skimming else None
    )
    masks, counts = prepared.destination_masks, prepared.destination_counts
    expected_masks = np.ones((1, 4)) if skimming else [[0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    np.testing.assert_array_equal(masks, expected_masks)
    np.testing.assert_array_equal(counts, [4] if skimming else [2, 1, 0, 0])
    assert not masks.flags.writeable and not counts.flags.writeable

    outputs = prepared.make_outputs()
    assert prepared.run(outputs) is outputs
    loads, skims = outputs.link_loads, outputs.skims
    first = np.array([7, 0, 12, 0])[:, None] * [1, 10]
    np.testing.assert_array_equal(loads, first)
    assert outputs.total_turn_penalty == (27.5 if turn else 0)
    if skimming:
        np.testing.assert_array_equal(skims[0, 3], [2.5 if turn else 2, 40])
        np.testing.assert_array_equal(skims[3, 3], 0)  # Isolated origin is processed.
        assert np.isinf(skims[3, :3]).all()
        assert not skims.flags.writeable
    else:
        assert skims is None
    assert not loads.flags.writeable
    snapshot = outputs.copy()
    history = outputs.copy()
    history_loads, history_skims = history.link_loads, history.skims
    assert not np.shares_memory(snapshot.link_loads, loads)
    if skimming:
        assert not np.shares_memory(snapshot.skims, skims)

    # Fixed inputs were copied, not retained as writable aliases.
    demand.fill(0)
    distance.fill(-999)
    changed_costs = np.array([5.0, 1.0, 1.0, 1.0])
    prepared.update_costs(changed_costs)
    assert np.shares_memory(prepared.costs, changed_costs)
    del changed_costs  # The memoryview retains the caller's allocation.
    np.testing.assert_array_equal(context.costs, 1)

    # No NumPy construction or setup helper may run in repeated iterations.
    # NumPy views are allowed; data allocations and preparation are not.
    def unexpected(*args, **kwargs):
        raise AssertionError("iteration attempted to prepare or allocate")

    with monkeypatch.context() as patch:
        for name in ("array", "empty", "zeros", "full", "tile", "count_nonzero"):
            patch.setattr(np, name, unexpected)
        patch.setattr("aequilibrae.paths.cython.aon_context.make_destination_masks", unexpected)
        patch.setattr("aequilibrae.paths.cython.aon_context.copy_input", unexpected)
        for _ in range(3):
            assert prepared.run(outputs) is outputs
    np.testing.assert_array_equal(loads, np.array([0, 7, 12, 7])[:, None] * [1, 10])
    assert outputs.total_turn_penalty == 0
    if skimming:
        # Costs changed the selected path, not the fixed cost skim's link values.
        np.testing.assert_array_equal(skims[0, 3], [3, 90])
        np.testing.assert_array_equal(snapshot.skims[0, 3], [2.5 if turn else 2, 40])
    np.testing.assert_array_equal(snapshot.link_loads, first)
    assert snapshot.total_turn_penalty == (27.5 if turn else 0)
    np.testing.assert_array_equal(masks, expected_masks)
    assert np.shares_memory(masks, prepared.destination_masks)
    assert np.shares_memory(counts, prepared.destination_counts)

    assert outputs.copy_to(history) is history
    np.testing.assert_array_equal(history.link_loads, loads)
    assert history.total_turn_penalty == 0
    del prepared, outputs, context, snapshot, history
    # Retained live and snapshot views keep their own allocations alive.
    np.testing.assert_array_equal(history_loads, loads)
    if skimming:
        np.testing.assert_array_equal(history_skims, skims)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("zones", [1, 2])
@pytest.mark.parametrize("skimming", [False, True])
def test_prepared_empty_and_edgeless(turn, zones, skimming):
    context = (TurnBasedContext if turn else NodeBasedContext)([0, 0, 0], [], [])
    prepared = PreparedAoN(context, np.ones((zones, zones, 1)), costs=context.costs,
                           cores=3, skim_fields=[np.empty(0)] if skimming else [])
    prepared.update_costs(np.empty(0))
    outputs = prepared.make_outputs()
    prepared.run(outputs)
    assert outputs.link_loads.shape == (0, 1)
    assert outputs.total_turn_penalty == 0
    if skimming:
        expected = np.full((zones, zones), np.inf)
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(outputs.skims[:, :, 0], expected)
    assert prepared.run(outputs) is outputs


@pytest.mark.parametrize("turn", [False, True])
def test_prepared_targets_do_not_cancel_demand_classes(turn):
    prepared = PreparedAoN(
        prepared_context(turn),
        np.array(
            [
                [[0, 0], [5, -5]],
                [[0, 0], [0, 0]],
            ]
        ),
        costs=np.ones(4),
        cores=2,
    )
    np.testing.assert_array_equal(prepared.origins, [0])
    np.testing.assert_array_equal(prepared.destination_masks, [[0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(prepared.destination_counts, [1, 0])
    outputs = prepared.make_outputs()
    np.testing.assert_array_equal(prepared.run(outputs).link_loads, [[5, -5], [0, 0], [0, 0], [0, 0]])


@pytest.mark.parametrize("turn", [False, True])
def test_prepared_context_can_be_shared_with_independent_costs(turn):
    from concurrent.futures import ThreadPoolExecutor

    context = prepared_context(turn)
    demand = np.zeros((4, 4, 1))
    demand[0, 3] = 10
    first, second = [PreparedAoN(context, demand, costs=context.costs, cores=2) for _ in range(2)]
    second.update_costs(np.array([5., 1., 1., 1.]))
    outputs = [first.make_outputs(), second.make_outputs()]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(lambda pair: pair[0].run(pair[1]), zip([first, second], outputs, strict=True))) == outputs
    np.testing.assert_array_equal(outputs[0].link_loads[:, 0], [10, 0, 10, 0])
    np.testing.assert_array_equal(outputs[1].link_loads[:, 0], [0, 10, 10, 10])
    np.testing.assert_array_equal(context.costs, 1)


def test_prepared_graph_helper_without_legacy_buffers():
    graph = history_graph(0.5)
    matrix = make_matrix(graph, classes=1)
    matrix.matrix_view[0, 3] = 10
    prepared = prepare_aon(matrix, graph, cores=2)
    assert np.shares_memory(prepared.costs, graph.compact_cost)
    snapshot, outputs = prepared.make_outputs(), prepared.make_outputs()
    prepared.run(snapshot)
    np.testing.assert_array_equal(snapshot.link_loads[:, 0], [10, 0, 10, 0])
    assert snapshot.total_turn_penalty == 5
    np.testing.assert_array_equal(snapshot.skims[0, 3], [2.5, 40])
    graph.compact_cost[0] = 5
    graph.compact_skims.fill(-999)
    matrix.matrix_view.fill(0)
    prepared.run(outputs)  # In-place cost edits are already visible; no rebind needed.
    np.testing.assert_array_equal(outputs.link_loads[:, 0], [0, 10, 10, 10])
    np.testing.assert_array_equal(outputs.skims[0, 3], [3, 90])
    assert outputs.total_turn_penalty == 0
    np.testing.assert_array_equal(snapshot.link_loads[:, 0], [10, 0, 10, 0])


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"cores": 0}, ValueError, "cores"),
        ({"cores": 1.5}, TypeError, "integer"),
        ({"skim_fields": [[1]]}, ValueError, "skim field"),
        ({"skim_penalties": [True]}, ValueError, "skim_penalties"),
        ({"origins": [0, 0]}, ValueError, "unique"),
        ({"origins": [-1]}, ValueError, "centroid"),
        ({"origins": [4]}, ValueError, "centroid"),
        ({"origins": [0.5]}, ValueError, "centroid"),
        ({"origins": [[0]]}, ValueError, "centroid"),
    ],
)
def test_prepared_setup_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        PreparedAoN(prepared_context(False), np.ones((4, 4, 1)), costs=np.ones(4), **kwargs)


@pytest.mark.parametrize("shape", [(4, 4), (4, 3, 1), (5, 5, 1), (4, 4, 0), (0, 0, 1)])
def test_prepared_demand_validation(shape):
    with pytest.raises(ValueError, match="demand"):
        PreparedAoN(prepared_context(False), np.zeros(shape), costs=np.ones(4))


def test_prepared_cost_update_and_snapshot_validation():
    context = prepared_context(True)
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs)
    outputs = prepared.make_outputs()
    snapshot = prepared.run(outputs).copy()
    for costs in ([1], [-1, 1, 1, 1], [np.nan, 1, 1, 1]):
        with pytest.raises(ValueError, match="cost"):
            prepared.update_costs(np.array(costs, dtype=np.float64))
        np.testing.assert_array_equal(prepared.run(outputs).link_loads, snapshot.link_loads)
    with pytest.raises(ValueError, match="turn restrictions"):
        PreparedAoN(context, demand, costs=context.costs, block_centroids=True)
    with pytest.raises(TypeError, match="context"):
        PreparedAoN(None, demand, costs=context.costs)
    with pytest.raises(TypeError, match="AoNOutputs"):
        snapshot.copy_to(None)
    for incompatible in (AoNOutputs(5, 4, 1, 0), AoNOutputs(4, 4, 1, 1)):
        with pytest.raises(ValueError, match="shapes"):
            snapshot.copy_to(incompatible)
        np.testing.assert_array_equal(incompatible.link_loads, 0)
    prepared.update_costs(np.full(4, np.inf))
    np.testing.assert_array_equal(prepared.run(outputs).link_loads, 0)
    assert outputs.total_turn_penalty == 0


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_prepared_cost_buffers_are_borrowed_and_rebound(turn, readonly, monkeypatch):
    context = prepared_context(turn)
    demand = np.zeros((4, 4, 1))
    demand[0, 3] = 10
    costs = np.array([5., 1., 1., 1.])
    costs.flags.writeable = not readonly
    original_ref = weakref.ref(costs)
    prepared = PreparedAoN(context, demand, costs=costs)
    outputs = prepared.make_outputs()
    original_view = prepared.costs
    assert np.shares_memory(original_view, costs)
    assert original_view.ctypes.data == costs.ctypes.data
    np.testing.assert_array_equal(prepared.run(outputs).link_loads[:, 0], [0, 10, 10, 10])
    if not readonly:
        costs[0] = 1  # Between-run mutations are visible without update_costs.
        np.testing.assert_array_equal(prepared.run(outputs).link_loads[:, 0], [10, 0, 10, 0])

    replacement = np.ones(4)
    replacement_ref = weakref.ref(replacement)

    def unexpected(*args, **kwargs):
        raise AssertionError("cost binding must not copy data")

    with monkeypatch.context() as patch:
        patch.setattr(np, "copyto", unexpected)
        patch.setattr(np, "array", unexpected)
        prepared.update_costs(memoryview(replacement).toreadonly())
    assert prepared.costs.ctypes.data == replacement.ctypes.data
    assert replacement.flags.writeable  # Binding does not alter the caller's flags.
    del costs, replacement
    gc.collect()
    assert original_ref() is not None  # The previously returned view still pins it.
    assert replacement_ref() is not None  # The new memoryview attribute pins it.
    del original_view
    gc.collect()
    assert original_ref() is None  # No stale owner or pointer binding remains.
    np.testing.assert_array_equal(prepared.run(outputs).link_loads[:, 0], [10, 0, 10, 0])
    del prepared
    gc.collect()
    assert replacement_ref() is None


@pytest.mark.parametrize("costs", [
    None, [1.] * 4, np.ones(4, dtype=np.float32), np.ones(4, dtype=np.int64),
    np.ones((2, 2)), np.ones(8)[::2], np.ones(4)[::-1], np.ones(3),
    np.array([np.nan, 1, 1, 1]), np.array([-1., 1, 1, 1]),
    np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1),
])
def test_prepared_rejects_invalid_cost_buffers_without_rebinding(costs):
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs)
    pointer = prepared.costs.ctypes.data
    for bind in (prepared.update_costs,
                 lambda value: PreparedAoN(context, np.ones((4, 4, 1)), costs=value)):
        with pytest.raises((TypeError, ValueError)):
            bind(costs)
        assert prepared.costs.ctypes.data == pointer
        np.testing.assert_array_equal(prepared.costs, 1)


def test_prepared_accepts_cython_cost_buffer_and_iterable_skims():
    context = prepared_context(False)
    costs = allocate_array["double"](4, True, 1)
    demand = np.zeros((4, 4, 1))
    demand[0, 3] = 10
    prepared = PreparedAoN(context, demand, costs=costs, skim_fields=(context.costs for _ in range(1)))
    assert prepared.costs.ctypes.data == np.asarray(costs).ctypes.data
    np.asarray(costs)[0] = 5
    outputs = prepared.run(prepared.make_outputs())
    np.testing.assert_array_equal(outputs.link_loads[:, 0], [0, 10, 10, 10])
    assert outputs.skims[0, 3, 0] == 3


def test_prepared_requires_explicit_costs():
    with pytest.raises(TypeError, match="costs"):
        PreparedAoN(prepared_context(False), np.ones((4, 4, 1)))


@pytest.mark.parametrize("skimming", [False, True])
def test_prepared_rejects_costs_aliasing_outputs_before_writes(skimming):
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs,
                           skim_fields=[context.costs] if skimming else [])
    outputs = prepared.run(prepared.make_outputs())
    costs = outputs.skims[0, :, 0] if skimming else outputs.link_loads[:, 0]
    prepared.update_costs(costs)
    snapshot = outputs.copy()
    with pytest.raises(ValueError, match="overlap output"):
        prepared.run(outputs)
    np.testing.assert_array_equal(outputs.link_loads, snapshot.link_loads)
    if skimming:
        np.testing.assert_array_equal(outputs.skims, snapshot.skims)
    with pytest.raises(ValueError, match="overlap worker scratch"):
        prepared.update_costs(prepared.thread_outputs[0][0, :, 0])


def test_prepared_explicit_origins_and_empty_target_masks():
    context = prepared_context(False)
    demand = np.zeros((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, origins=[0])
    np.testing.assert_array_equal(prepared.destination_counts, 0)
    np.testing.assert_array_equal(prepared.run(prepared.make_outputs()).link_loads, 0)
    prepared = PreparedAoN(context, demand, costs=context.costs, origins=[0], skim_fields=[context.costs])
    outputs = prepared.make_outputs()
    prepared.run(outputs)
    np.testing.assert_array_equal(outputs.skims[0, :, 0], [0, 1, 1, 2])
    assert np.isinf(outputs.skims[1:]).all()


@pytest.mark.parametrize("origins", [[], [2, 0]])
def test_prepared_origin_subsets(origins):
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs,
                           origins=origins, skim_fields=[context.costs])
    np.testing.assert_array_equal(prepared.origins, sorted(origins))
    outputs = prepared.run(prepared.make_outputs())
    for origin in range(4):
        if origin in origins:
            assert outputs.skims[origin, origin, 0] == 0
        else:
            assert np.isinf(outputs.skims[origin]).all()
    if not origins:
        np.testing.assert_array_equal(outputs.link_loads, 0)


def test_prepared_turn_totals_accumulate_across_origins():
    context = TurnBasedContext([0, 1, 2, 3, 3], [1, 2, 3], [1.0] * 3, [0, 0, 1, 1], [2], [0.5])
    demand = np.zeros((4, 4, 1))
    demand[0, 3, 0], demand[1, 3, 0] = 2, 3
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=1, skim_fields=[context.costs])
    outputs = prepared.make_outputs()
    # Every element of multidimensional allocations must be initialized.
    np.testing.assert_array_equal(outputs.link_loads, 0)
    assert np.isinf(outputs.skims).all()
    for array in (outputs.link_loads, outputs.skims, prepared.origins,
                  prepared.destination_masks, prepared.destination_counts):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    for _ in range(2):
        prepared.run(outputs)
        np.testing.assert_array_equal(outputs.link_loads[:, 0], [2, 5, 5])
        assert outputs.total_turn_penalty == 2.5


def test_prepared_reuses_node_centroid_blocking_heads():
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs,
                           cores=3, skim_fields=[context.costs], block_centroids=True)
    outputs = prepared.make_outputs()
    for cost in (1, 5, 2):
        prepared.update_costs(np.full(4, cost, dtype=np.float64))
        prepared.run(outputs)
        np.testing.assert_array_equal(outputs.link_loads, 1)
        np.testing.assert_array_equal(outputs.skims[[0, 0, 1, 2], [1, 2, 3, 1], 0], 1)
        assert np.isinf(outputs.skims[0, 3, 0])
        assert np.isinf(outputs.skims[2, 3, 0])
        np.testing.assert_array_equal(context.heads, [1, 2, 3, 1])


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("skimming", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
def test_prepared_rotates_outputs_without_copying(turn, skimming, cores, monkeypatch):
    context = prepared_context(turn)
    demand = np.zeros((4, 4, 1))
    demand[0, 3] = 5
    prepared = PreparedAoN(
        context,
        demand,
        costs=context.costs,
        cores=cores,
        skim_fields=[context.costs] if skimming else [],
        skim_penalties=[True] if skimming else None,
    )
    slots = [prepared.make_outputs() for _ in range(3)]
    views = [(out.link_loads, out.skims) for out in slots]
    states = [None] * len(slots)
    expected_loads = ([5, 0, 5, 0], [0, 5, 5, 5])

    def unexpected(*args, **kwargs):
        raise AssertionError("run attempted an output copy or allocation")

    for iteration in range(8):
        slot, state = iteration % len(slots), iteration % 2
        prepared.update_costs(np.array([5 if state else 1, 1, 1, 1], dtype=np.float64))
        with monkeypatch.context() as patch:
            for name in ("array", "empty", "zeros", "full", "tile", "copyto"):
                patch.setattr(np, name, unexpected)
            assert prepared.run(slots[slot]) is slots[slot]
        states[slot] = state
        for index, out in enumerate(slots):
            loads, skims = views[index]
            assert np.shares_memory(loads, out.link_loads)
            if skimming:
                assert np.shares_memory(skims, out.skims)
            previous_state = states[index]
            if previous_state is None:
                np.testing.assert_array_equal(out.link_loads, 0)
                if skimming:
                    assert np.isinf(out.skims).all()
                continue
            # Every other buffer retains its previous iteration, including scalars.
            np.testing.assert_array_equal(out.link_loads[:, 0], expected_loads[previous_state])
            assert out.total_turn_penalty == (2.5 if turn and not previous_state else 0)
            if skimming:
                assert out.skims[0, 3, 0] == (3 if previous_state else 2.5 if turn else 2)


@pytest.mark.parametrize("fields", [0, 1])
def test_run_validates_output_layout_before_writes(fields):
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs, skim_fields=[context.costs] * fields)
    layout = (4, 4, 1, fields)
    # Outputs need not come from this assignment; matching fixed layouts suffice.
    out = AoNOutputs(*layout)
    assert prepared.run(out) is out
    for axis in range(4):
        incompatible = list(layout)
        incompatible[axis] += 1
        target = AoNOutputs(*incompatible)
        with pytest.raises(ValueError, match="shape does not match"):
            prepared.run(target)
        np.testing.assert_array_equal(target.link_loads, 0)
        assert target.total_turn_penalty == 0
        if target.skims is not None:
            assert np.isinf(target.skims).all()
    for target in (None, out.link_loads, object()):
        with pytest.raises(TypeError):
            prepared.run(target)
    with pytest.raises(TypeError):
        prepared.run()
    with pytest.raises(AttributeError):
        out.link_loads_buffer = np.zeros((4, 1))
    with pytest.raises(RuntimeError, match="reinitialized"):
        out.__init__(*layout)
    with pytest.raises(RuntimeError, match="reinitialized"):
        prepared.__init__(context, np.ones((4, 4, 1)), costs=context.costs)
    # copy() preserves even the zone dimension of loading-only layouts.
    copied = out.copy()
    assert prepared.run(copied) is copied
    assert prepared.run(out) is out


def test_output_guard_released_on_iteration_failure(monkeypatch):
    prepared = PreparedAoN(prepared_context(False), np.ones((4, 4, 1)), costs=np.ones(4))
    other = PreparedAoN(prepared_context(False), np.ones((4, 4, 1)), costs=np.ones(4))
    out = prepared.make_outputs()

    def failed_reduction(*args, **kwargs):
        raise RuntimeError("injected reduction failure")

    with monkeypatch.context() as patch:
        patch.setattr(np, "sum", failed_reduction)
        with pytest.raises(RuntimeError, match="injected"):
            prepared.run(out)
    assert prepared.run(out) is out
    assert other.run(out) is out


def select_link_reference(context, demand, selections, origins=None):
    """Walk each OD path so the test does not repeat the kernel's tree logic."""
    from aequilibrae.paths.cython.dijkstra import dijkstra

    zones, _, classes = demand.shape
    loads = np.zeros((len(selections), context.link_count, classes))
    od = np.zeros((len(selections), zones, zones, classes))
    results = context.make_results()
    for origin in range(zones) if origins is None else origins:
        dijkstra(context, origin, range(zones), results)
        for destination in range(zones):
            path = results.path_links_to(destination).tolist()
            for selection, members in enumerate(selections.values()):
                if set(path).intersection(members):
                    od[selection, origin, destination] = demand[origin, destination]
                    for link in path:
                        loads[selection, link] += demand[origin, destination]
    return loads, od


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
def test_select_link_history_rotation_and_ownership(turn, cores, monkeypatch):
    args = ([0, 2, 3, 4, 4], [1, 2, 3, 1], [1.] * 4)
    context = TurnBasedContext(*args, [0, 1, 1, 1, 1], [2], [10.]) if turn else NodeBasedContext(*args)
    demand = np.zeros((4, 4, 2))
    demand[0] = [[999, 999], [2, -2], [3, 30], [5, 50]]
    demand[1, 3] = [7, 70]
    demand[3, 0] = [11, 110]  # Unreachable.
    selected = {"direct": [0], "history": [3], "both": [0, 1, 2, 2], "empty": []}
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=cores,
                           selected_links=selected, skim_fields=[context.costs])
    assert prepared.select_link_names == tuple(selected)
    expected = select_link_reference(context, demand, selected)
    selected["direct"].clear()  # Input membership is snapshotted.
    first, second = prepared.make_outputs(), prepared.make_outputs()
    assert prepared.run(first) is first
    for actual, reference in zip((first.select_link_loads, first.select_link_od), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    # A matched route includes upstream and downstream links, counted only once
    # even when multiple selected members occur on it.
    np.testing.assert_array_equal(first.select_link_loads[2], first.link_loads)
    if turn:
        np.testing.assert_array_equal(first.select_link_od[1, 0, 1], 0)
        np.testing.assert_array_equal(first.select_link_od[1, 0, 3], [5, 50])
        np.testing.assert_array_equal(first.select_link_loads[1, [1, 2, 3]], [[5, 50]] * 3)
    views = first.select_link_loads, first.select_link_od
    for view in (*views, prepared.select_link_masks, prepared.thread_select_link_loads):
        with pytest.raises(ValueError):
            view.flags.writeable = True
    snapshot = first.copy()
    assert snapshot.select_link_names == first.select_link_names
    assert not np.shares_memory(snapshot.select_link_loads, views[0])
    prepared.update_costs(np.array([5., 1., 1., 1.]))

    def unexpected(*args, **kwargs):
        raise AssertionError("select-link run allocated/copied/prepared numeric buffers")

    with monkeypatch.context() as patch:
        for name in ("array", "empty", "zeros", "full", "tile", "copyto"):
            patch.setattr(np, name, unexpected)
        patch.setattr("aequilibrae.paths.cython.aon_context.make_select_link_masks", unexpected)
        for _ in range(3):
            prepared.run(second)
    np.testing.assert_array_equal(first.select_link_loads, expected[0])
    np.testing.assert_array_equal(first.select_link_od, expected[1])
    np.testing.assert_array_equal(second.select_link_loads[0], 0)
    np.testing.assert_array_equal(second.select_link_od[0], 0)
    assert second.copy_to(first) is first
    np.testing.assert_array_equal(views[0], second.select_link_loads)
    np.testing.assert_array_equal(views[1], second.select_link_od)
    del prepared, first, second, context
    gc.collect()
    np.testing.assert_array_equal(views[0][0], 0)
    np.testing.assert_array_equal(snapshot.select_link_loads, expected[0])


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_select_link_random_multigraph_path_walks(turn, seed):
    rng = np.random.default_rng(seed)
    nodes, zones, links = 9, 6, 32
    tails = np.sort(rng.integers(nodes, size=links))
    heads = rng.integers(nodes, size=links)
    fs = np.r_[0, np.cumsum(np.bincount(tails, minlength=nodes))]
    costs = rng.integers(0, 5, size=links).astype(float)
    kwargs = {}
    if turn:
        turn_fs, to_links, penalties = [0], [], []
        for head in heads:
            for outgoing in range(fs[head], fs[head + 1]):
                if rng.random() < 0.4:
                    to_links.append(outgoing)
                    penalties.append(rng.choice([0., 2., 10., np.inf]))
            turn_fs.append(len(to_links))
        kwargs = {"turn_fs": turn_fs, "turn_to_links": to_links, "turn_penalties": penalties,
                  "allow_uturns": False}
    context = (TurnBasedContext if turn else NodeBasedContext)(fs, heads, costs, **kwargs)
    demand = rng.integers(-3, 8, size=(zones, zones, 3)).astype(float)
    selections = {str(i): rng.choice(links, size=i * 3).tolist() for i in range(5)}
    origins = [0, 2, 5]
    expected = select_link_reference(context, demand, selections, origins)
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=3,
                           selected_links=selections, origins=origins)
    outputs = prepared.make_outputs()
    for _ in range(2):
        prepared.run(outputs)
        np.testing.assert_array_equal(outputs.select_link_loads, expected[0])
        np.testing.assert_array_equal(outputs.select_link_od, expected[1])


@pytest.mark.parametrize("turn", [False, True])
def test_single_origin_select_link_partial_empty_and_scratch(turn):
    from aequilibrae.paths.cython.dijkstra import dijkstra

    context = prepared_context(turn)
    results = context.make_results()
    demand = np.ones((4, 2))
    demand.flags.writeable = False
    loads, od = np.full((4, 2), 7.), np.full((4, 2), 99.)
    returned = results.select_link_loading([0], demand, loads, od)
    assert returned[0] is loads and returned[1] is od
    np.testing.assert_array_equal(loads, 7)
    np.testing.assert_array_equal(od, 0)
    dijkstra(context, 0, 1, results)  # Destination 3 is unfinalized.
    results.select_link_loading([0, 0], demand, loads, od)
    np.testing.assert_array_equal(loads, [[8, 8], [7, 7], [7, 7], [7, 7]])
    np.testing.assert_array_equal(od, [[0, 0], [1, 1], [0, 0], [0, 0]])
    scratch, flags = results.workspace.state_loads, results.workspace.selected_paths
    results.select_link_loading([], demand, loads, od)
    np.testing.assert_array_equal(od, 0)
    np.testing.assert_array_equal(scratch, 0)
    np.testing.assert_array_equal(flags, False)
    assert np.shares_memory(scratch, results.workspace.state_loads)
    assert np.shares_memory(flags, results.workspace.selected_paths)
    # Destination and class axes can independently be empty.
    results.select_link_loading([0], np.empty((0, 2)), loads, np.empty((0, 2)))
    results.select_link_loading([0], np.empty((4, 0)), np.empty((4, 0)), np.empty((4, 0)))
    np.testing.assert_array_equal(scratch, 0)  # Old allocation survives resizing.
    with pytest.raises(ValueError):
        flags.flags.writeable = True


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("origins", [None, []])
def test_select_link_edgeless_and_no_active_origins(turn, origins):
    context = (TurnBasedContext if turn else NodeBasedContext)([0, 0, 0], [], [])
    prepared = PreparedAoN(context, np.ones((2, 2, 1)), costs=context.costs, cores=3,
                           selected_links={"empty": []}, origins=origins)
    outputs = prepared.run(prepared.make_outputs())
    assert outputs.select_link_loads.shape == (1, 0, 1)
    np.testing.assert_array_equal(outputs.select_link_od, 0)
    prepared.run(outputs)
    np.testing.assert_array_equal(outputs.select_link_od, 0)


@pytest.mark.parametrize("selected, error", [
    ([0], TypeError), ({"x": [-1]}, ValueError), ({"x": [4]}, ValueError),
    ({"x": [0.5]}, TypeError), ({"x": [True]}, TypeError), ({"x": [[0]]}, TypeError),
])
def test_select_link_setup_validation(selected, error):
    context = prepared_context(False)
    with pytest.raises(error):
        PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs, selected_links=selected)


def test_select_link_output_layout_and_cost_alias_validation():
    context = prepared_context(False)
    prepared = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs,
                           selected_links={"x": [0], "y": [1]})
    out = prepared.run(prepared.make_outputs())
    for names in ((), ("x",), ("y", "x"), ("x", "z")):
        other = AoNOutputs(*out.shape, select_link_names=names)
        with pytest.raises(ValueError, match="shape"):
            prepared.run(other)
        with pytest.raises(ValueError, match="shapes"):
            out.copy_to(other)
        np.testing.assert_array_equal(other.link_loads, 0)
    for costs in (out.select_link_loads[0, :, 0], out.select_link_od[0, 0, :, 0]):
        prepared.update_costs(costs)
        snapshot = out.copy()
        with pytest.raises(ValueError, match="overlap output"):
            prepared.run(out)
        np.testing.assert_array_equal(out.select_link_loads, snapshot.select_link_loads)
        np.testing.assert_array_equal(out.select_link_od, snapshot.select_link_od)
    with pytest.raises(ValueError, match="worker scratch"):
        prepared.update_costs(prepared.thread_select_link_loads[0, 0, :, 0])
    with pytest.raises(ValueError, match="unique"):
        AoNOutputs(*out.shape, select_link_names=("x", "x"))
    disabled = PreparedAoN(context, np.ones((4, 4, 1)), costs=context.costs)
    assert disabled.thread_select_link_loads is None
    assert disabled.select_link_masks.shape == (0, context.link_count)
    plain = disabled.make_outputs()
    assert plain.select_link_loads is None and plain.select_link_od is None


def test_single_origin_select_link_validation_and_nonfinite_demand():
    from aequilibrae.paths.cython.dijkstra import dijkstra

    context = prepared_context(False)
    results = dijkstra(context, 0, None)
    demand = np.zeros((4, 2))
    demand[1] = [np.nan, np.inf]
    loads, od = np.zeros((4, 2)), np.empty((4, 2))
    results.select_link_loading([0], demand, loads, od)
    assert np.isnan(loads[0, 0]) and np.isposinf(loads[0, 1])
    np.testing.assert_array_equal(loads[1:], 0)
    np.testing.assert_array_equal(od[1], demand[1])
    for invalid in ([-1], [4], [True], [1.5]):
        with pytest.raises((ValueError, TypeError)):
            results.select_link_loading(invalid, demand, loads, od)
    for bad_demand in (demand.tolist(), demand.astype(np.float32), demand[:, ::-1], np.ones((5, 2))):
        with pytest.raises((ValueError, TypeError)):
            results.select_link_loading([0], bad_demand, loads, od)
    for bad_loads, bad_od in ((None, od), (loads, None), (loads, loads), (demand, od),
                              (loads, demand), (loads[:, ::-1], od), (loads, od[:2])):
        with pytest.raises((ValueError, TypeError)):
            results.select_link_loading([0], demand, bad_loads, bad_od)
    scratch = results.workspace.state_loads
    with pytest.raises(ValueError, match="overlap"):
        results.select_link_loading([0], scratch, loads, od)


@pytest.mark.parametrize("penalty", [None, 0.5, 10., np.inf])
@pytest.mark.parametrize("cores", [1, 3])
def test_select_link_legacy_adapter(penalty, cores):
    graph = history_graph(penalty)
    matrix = make_matrix(graph)
    matrix.matrix_view[0] = [[999, 999], [2, 20], [3, 30], [5, 50]]
    matrix.matrix_view[1, 3] = [7, 70]
    selected = {"direct": [0], "history": [3], "both": [0, 1, 2], "empty": []}

    def selected_buffers():
        result = AssignmentResults()
        result.set_cores(cores)
        result._selected_links = {name: np.array(links, dtype=np.int64) for name, links in selected.items()}
        result.prepare(graph, matrix)
        aux = MultiThreadedAoN()
        aux.prepare(graph, result)
        return result, aux

    old, old_aux = selected_buffers()
    new, new_aux = selected_buffers()
    old_report = aon_parallel(matrix, graph, old, old_aux, cores)
    assert aon_parallel_context(matrix, graph, new, new_aux, cores) == old_report
    np.testing.assert_allclose(new_aux.temp_sl_od_matrix.sum(axis=0), old_aux.temp_sl_od_matrix.sum(axis=0))
    np.testing.assert_allclose(new_aux.temp_sl_link_loading.sum(axis=0), old_aux.temp_sl_link_loading.sum(axis=0))
    np.testing.assert_allclose(new_aux.temp_link_loads.sum(axis=0), old_aux.temp_link_loads.sum(axis=0))
    prepared = prepare_aon(matrix, graph, cores=cores, selected_links=selected)
    expected = prepared.run(prepared.make_outputs())
    np.testing.assert_array_equal(new_aux.temp_sl_od_matrix.sum(axis=0), expected.select_link_od)
    # OD rows replace; loads accumulate, matching the adapter's existing contract.
    aon_parallel_context(matrix, graph, new, new_aux, cores)
    np.testing.assert_array_equal(new_aux.temp_sl_od_matrix.sum(axis=0), expected.select_link_od)
    np.testing.assert_array_equal(new_aux.temp_sl_link_loading.sum(axis=0), expected.select_link_loads * 2)
    # Clear stale OD rows even if a later call uses fewer than the allocated workers.
    new_aux.temp_sl_od_matrix[:, :, :3] = 99
    aon_parallel_context(matrix, graph, new, new_aux, 1)
    np.testing.assert_array_equal(new_aux.temp_sl_od_matrix.sum(axis=0), expected.select_link_od)
    before = new_aux.temp_link_loads.copy()
    new_aux.select_links[0, 0] = graph.compact_num_links
    with pytest.raises(ValueError, match="valid compact"):
        aon_parallel_context(matrix, graph, new, new_aux, cores)
    np.testing.assert_array_equal(new_aux.temp_link_loads, before)
