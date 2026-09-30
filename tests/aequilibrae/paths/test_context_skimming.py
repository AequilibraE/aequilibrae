"""Skimming owners work independently and use the same operation as assignment."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.network_loading import sum_weighted_turn_costs
from aequilibrae.paths.cython.outputs import SkimmingOutputs
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.skimming import skimming
from aequilibrae.paths.cython.context import SkimmingContext
from aequilibrae.paths.cython.workspaces import SkimmingWorkspace
from .routing_helpers import allocate_results, history_context, make_context, path_walk_outputs, search


def workspace_for(context, inputs):
    return SkimmingWorkspace(context.state_count, inputs.additive_field_count) if inputs.additive_field_count else None


def path_walk_skims(results, inputs, destination_count, cost_name=None, turn_cost_name=None):
    """Walk each destination separately rather than using the state cascade."""
    expected = np.full((inputs.field_count, destination_count), np.inf)
    fields = inputs.fields
    for node in range(destination_count):
        if not results.reachable_to(node):
            continue
        links = results.path_links_to(node)
        for field, name in enumerate(inputs.field_names):
            if name == cost_name:
                expected[field, node] = results.path_cost_to(node)
            elif name == turn_cost_name:
                expected[field, node] = results.path_turn_cost_to(node)
            else:
                expected[field, node] = fields[name][links].sum()
    return expected


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("penalty", [0.5, 10.0, np.inf])
@pytest.mark.parametrize("width", [0, 1, 3, 37])
def test_skim_state_tree_against_path_sums_and_assignment(turn, penalty, width):
    context = history_context(penalty, turn=turn)
    fields = [np.arange(context.link_count, dtype=np.float64) + i for i in range(width)]
    inputs = SkimmingContext(
        context.link_count,
        link_fields={f"field_{i}": field for i, field in enumerate(fields)},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    demand = np.ones((4, 4, 2))
    prepared = PreparedAoN(context, demand, skimming=inputs, cores=3)
    assigned = prepared.run(prepared.make_outputs())
    output = inputs.make_outputs(4, origin_count=4)
    scratch = workspace_for(context, inputs)
    results = allocate_results(context)
    for origin in range(4):
        search(context, origin, results=results)
        assert skimming(results, inputs, scratch, output, origin_row=origin) is output
        expected = path_walk_skims(results, inputs, 4, "objective", "penalty")
        np.testing.assert_allclose(output.skims[origin], expected)
    np.testing.assert_allclose(output.skims, assigned.skimming.skims)
    expected_fields = path_walk_outputs(context, demand, fields)[1]
    np.testing.assert_allclose(output.skims[:, :width], expected_fields)


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("turn_cost", [False, True])
def test_each_combination_of_skim_groups(plain, cost, turn_cost):
    context = history_context(0.5)
    link_fields = {"distance": np.ones(4), "toll": np.full(4, 2.0)} if plain else {}
    inputs = SkimmingContext(
        4,
        link_fields=link_fields,
        cost_name="cost" if cost else None,
        turn_cost_name="turns" if turn_cost else None,
    )
    output = inputs.make_outputs(3, origin_count=2)
    scratch = workspace_for(context, inputs)
    results = allocate_results(context)
    # Missing groups must not leave gaps or shift another group's matrices.
    for row, origin in enumerate((0, 2)):
        saved_other_row = output.skims[1 - row].copy()
        search(context, origin, results=results)
        skimming(results, inputs, scratch, output, origin_row=row)
        np.testing.assert_array_equal(output.skims[row], path_walk_skims(results, inputs, 3, "cost", "turns"))
        np.testing.assert_array_equal(output.skims[1 - row], saved_other_row)


def test_names_order_and_one_shot_allocation():
    context = history_context(0.5)
    inputs = SkimmingContext(
        4,
        link_fields={"distance": np.ones(4), "time": np.ones(4)},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    assert inputs.field_names == ("distance", "time", "objective", "penalty")
    assert inputs.additive_field_count == 2
    output = inputs.make_outputs(context.node_count)
    assert output.skims.shape == (1, 4, 4)
    assert output.skims.flags.c_contiguous
    assert np.all(np.isinf(output.skims))
    # A nonzero physical origin still writes row zero in a one-shot output.
    results = search(context, 2)
    skimming(results, inputs, workspace_for(context, inputs), output)
    matrices = output.matrices
    assert list(matrices) == list(inputs.field_names)
    for field, name in enumerate(inputs.field_names):
        assert matrices[name].shape == (1, 4)
        assert np.shares_memory(matrices[name], output.skims)
        np.testing.assert_array_equal(matrices[name], output.skims[:, field, :])
        assert matrices[name].flags.c_contiguous
        assert not matrices[name].flags.writeable
        with pytest.raises(ValueError):
            matrices[name].flags.writeable = True
    assert np.all(output.skims[0, :, 2] == 0)
    assert np.all(np.isinf(output.skims[0, :, 0]))


def test_origin_major_layout_with_rectangular_od_matrices():
    context = history_context(0.5)
    inputs = SkimmingContext(
        4,
        link_fields={"distance": np.ones(4), "time": np.full(4, 2.0)},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    output = inputs.make_outputs(2, origin_count=3)
    scratch = workspace_for(context, inputs)
    assert output.skims.shape == (3, 4, 2)
    assert output.skims.flags.c_contiguous
    assert output.skims.strides == (4 * 2 * 8, 2 * 8, 8)
    for row in range(output.origin_count):
        origin_view = output.skims[row]
        assert origin_view.flags.c_contiguous
        assert np.shares_memory(origin_view, output.skims)
        assert origin_view.ctypes.data == output.skims.ctypes.data + row * origin_view.nbytes
    matrices = output.matrices
    for field, matrix in enumerate(matrices.values()):
        assert matrix.shape == (3, 2)
        assert matrix.strides == (4 * 2 * 8, 8)
        assert not matrix.flags.c_contiguous
        assert not matrix.flags.writeable
        with pytest.raises(ValueError):
            matrix.flags.writeable = True
        assert np.shares_memory(matrix, output.skims)
        assert matrix.ctypes.data == output.skims.ctypes.data + field * 2 * 8
    for row, origin in ((2, 0), (0, 2), (1, 3)):
        results = search(context, origin)
        skimming(results, inputs, scratch, output, origin_row=row)
        expected = path_walk_skims(results, inputs, 2, "objective", "penalty")
        np.testing.assert_array_equal(output.skims[row], expected)
        for field, matrix in enumerate(matrices.values()):
            np.testing.assert_array_equal(matrix[row], expected[field])
    # Retained strided matrices reflect resets without a second allocation.
    output.reset()
    assert all(np.all(np.isinf(matrix)) for matrix in matrices.values())


@pytest.mark.parametrize("turn", [False, True])
def test_borrowing_and_field_meanings_survive_objective_rebinding(turn):
    context = history_context(0.5, turn=turn)
    # Sharing the objective buffer does not turn a link field into a label field.
    field = context.costs
    inputs = SkimmingContext(
        4,
        link_fields={"links": field},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    assert np.shares_memory(inputs.fields["links"], field)
    results = search(context, 0)
    output = inputs.make_outputs(4)
    scratch = workspace_for(context, inputs)
    skimming(results, inputs, scratch, output)
    previous = output.skims.copy()
    context.update_costs(np.full(4, 2.0))
    skimming(results, inputs, scratch, output)
    np.testing.assert_array_equal(output.skims, previous)
    search(context, 0, results=results)
    skimming(results, inputs, scratch, output)
    np.testing.assert_array_equal(output.skims[0], path_walk_skims(results, inputs, 4, "objective", "penalty"))
    assert output.matrices["objective"][0, 1] == 2
    assert output.matrices["links"][0, 1] == 1


def test_input_updates_do_not_change_writeability_or_allocate_output():
    context = history_context()
    field = np.ones(4)
    inputs = SkimmingContext(4, link_fields={"distance": field})
    assert field.flags.writeable
    assert not inputs.fields["distance"].flags.writeable
    output = inputs.make_outputs(4)
    scratch = workspace_for(context, inputs)
    results = search(context, 0)
    retained_output, retained_scratch = output.skims, scratch.state_skims
    output_pointer = retained_output.ctypes.data
    scratch_pointer = retained_scratch.ctypes.data
    tree = results.predecessors.copy()
    for value in (1.0, 2.0, -3.0):
        field[:] = value
        skimming(results, inputs, scratch, output)
        assert retained_output[0, 0, 3] == 3 * value
        assert output.skims.ctypes.data == output_pointer
        assert scratch.state_skims.ctypes.data == scratch_pointer
        np.testing.assert_array_equal(results.predecessors, tree)
    output.reset()
    assert np.all(np.isinf(retained_output))
    assert retained_scratch[results.terminal_states[3], 0] == -9


def test_partial_and_presearch_replace_only_the_requested_row():
    context = history_context()
    inputs = SkimmingContext(
        4,
        link_fields={"links": np.ones(4), "time": np.ones(4)},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    scratch = workspace_for(context, inputs)
    output = inputs.make_outputs(4, origin_count=2)
    results = search(context, 0)
    skimming(results, inputs, scratch, output, origin_row=0)
    saved = output.skims[0].copy()
    search(context, 0, targets=[1], results=results)
    skimming(results, inputs, scratch, output, origin_row=1)
    np.testing.assert_array_equal(output.skims[0], saved)
    assert np.all(np.isinf(output.skims[1, :, 3]))
    unfinalized = np.ones(context.state_count, dtype=bool)
    unfinalized[results.settlement_order[: results.settled_count]] = False
    assert np.all(np.isinf(scratch.state_skims[unfinalized]))
    skimming(allocate_results(context), inputs, scratch, output, origin_row=1)
    assert np.all(np.isinf(output.skims[1]))
    assert np.all(np.isinf(scratch.state_skims))
    np.testing.assert_array_equal(output.skims[0], saved)


@pytest.mark.parametrize("turn", [False, True])
def test_centroid_skims_use_intermediate_states(turn):
    context = make_context([0, 1, 2, 4], [2, 2, 0, 1], [1, 1, 1, 1], turn=turn)
    inputs = SkimmingContext(4, link_fields={"distance": np.ones(4)})
    output = inputs.make_outputs(2, origin_count=2)
    scratch = workspace_for(context, inputs)
    for origin in range(2):
        skimming(search(context, origin), inputs, scratch, output, origin_row=origin)
    np.testing.assert_array_equal(output.matrices["distance"], [[0, 2], [2, 0]])


@pytest.mark.parametrize("turn", [False, True])
def test_zero_cost_cycle_and_nonfinite_link_fields(turn):
    context = make_context([0, 2, 3, 5, 5], [1, 2, 2, 1, 3], [0, 0, 0, 0, 1], turn=turn)
    inputs = SkimmingContext(
        5,
        link_fields={
            "signed": np.array([-2.0, -3.0, -1.0, -1.0, -4.0]),
            "nan": np.full(5, np.nan),
            "inf": np.full(5, np.inf),
        },
        cost_name="objective",
        turn_cost_name="penalty",
    )
    results = search(context, 0)
    output = inputs.make_outputs(4)
    skimming(results, inputs, workspace_for(context, inputs), output)
    np.testing.assert_allclose(
        output.skims[0], path_walk_skims(results, inputs, 4, cost_name="objective", turn_cost_name="penalty")
    )
    assert np.all(output.skims[0, :, 0] == 0)


def test_link_fields_and_turn_costs_can_be_added_by_the_caller():
    context = history_context(0.5)
    inputs = SkimmingContext(4, link_fields={"links": context.costs}, cost_name="cost", turn_cost_name="turns")
    output = inputs.make_outputs(4)
    results = search(context, 0)
    skimming(results, inputs, workspace_for(context, inputs), output)
    matrices = output.matrices

    np.testing.assert_array_equal(matrices["links"] + matrices["turns"], matrices["cost"])
    assert matrices["links"][0, 3] == 2.0
    assert matrices["turns"][0, 3] == 0.5
    assert matrices["cost"][0, 3] == 2.5


def test_cost_projection_preserves_labels_exactly():
    context = make_context([0, 1, 2, 3, 3], [1, 2, 3], [0.1, 0.2, 0.3], {(0, 1): 0.4, (1, 2): 0.5})
    inputs = SkimmingContext(3, cost_name="objective", turn_cost_name="penalty")
    results = search(context, 0)
    output = inputs.make_outputs(4)
    skimming(results, inputs, None, output)
    for node in range(4):
        assert output.matrices["objective"][0, node] == results.path_cost_to(node)
        assert output.matrices["penalty"][0, node] == results.path_turn_cost_to(node)


@pytest.mark.parametrize("turn", [False, True])
def test_zero_links_and_empty_dimensions(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    inputs = SkimmingContext(0, link_fields={"empty": np.empty(0)}, cost_name="objective")
    results = search(context, 1)
    scratch = workspace_for(context, inputs)
    output = inputs.make_outputs(2)
    skimming(results, inputs, scratch, output)
    assert np.all(output.skims[0, :, 1] == 0)
    assert np.all(np.isinf(output.skims[0, :, 0]))
    empty_destinations = inputs.make_outputs(0)
    skimming(results, inputs, scratch, empty_destinations)
    assert empty_destinations.skims.shape == (1, 2, 0)
    assert np.all(scratch.state_skims[results.root] == 0)
    no_fields = SkimmingContext(0)
    empty_fields = no_fields.make_outputs(2)
    skimming(results, no_fields, None, empty_fields)
    assert empty_fields.skims.shape == (1, 0, 2)
    assert empty_fields.matrices == {}
    no_rows = inputs.make_outputs(2, origin_count=0)
    assert no_rows.skims.shape == (0, 2, 2)
    for out in (empty_destinations, empty_fields, no_rows):
        out.reset()
    with pytest.raises(ValueError, match="origin_row"):
        skimming(results, inputs, scratch, no_rows)


@pytest.mark.parametrize(
    "bad", [np.ones(4, dtype=np.float32), np.ones(8)[::2], [1.0] * 4, np.ones((4, 1)), np.ones(3), None]
)
def test_invalid_input_buffers(bad):
    with pytest.raises((TypeError, ValueError)):
        SkimmingContext(4, link_fields={"bad": bad})


def test_removed_combined_field_option_is_rejected():
    with pytest.raises(TypeError):
        SkimmingContext(4, link_fields_with_turn_costs={"time": np.ones(4)})


def test_skim_snapshot_copies_named_matrices():
    values = np.arange(12.0).reshape(3, 4)
    fields = {"time": values, "distance": values[:, ::-1]}
    output = SkimmingOutputs.from_matrices(fields)

    assert output.field_names == ("time", "distance")
    assert output.skims.shape == (3, 2, 4)
    for name, source in fields.items():
        np.testing.assert_array_equal(output.matrices[name], source)
        assert not np.shares_memory(output.matrices[name], source)
        assert not output.matrices[name].flags.writeable

    values[:] = -1
    assert output.matrices["time"][0, 0] == 0


@pytest.mark.parametrize(
    "matrices",
    [{}, {"time": np.ones(4)}, {"time": np.ones((2, 3)), "distance": np.ones((3, 2))}],
)
def test_skim_snapshot_checks_matrix_shapes(matrices):
    with pytest.raises(ValueError):
        SkimmingOutputs.from_matrices(matrices)


def test_unaligned_and_readonly_inputs():
    unaligned = np.ndarray((4,), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        SkimmingContext(4, link_fields={"bad": unaligned})
    field = np.ones(4)
    field.flags.writeable = False
    inputs = SkimmingContext(4, link_fields={"readonly": field})
    assert not field.flags.writeable
    assert np.shares_memory(inputs.fields["readonly"], field)


@pytest.mark.parametrize(
    "options",
    [
        {"link_fields": {"same": np.ones(4)}, "cost_name": "same"},
        {"link_fields": {"same": np.ones(4)}, "turn_cost_name": "same"},
        {"cost_name": "same", "turn_cost_name": "same"},
        {"cost_name": ""},
        {"turn_cost_name": 1},
        {"link_fields": {1: np.ones(4)}},
        {"link_fields": [np.ones(4)]},
    ],
)
def test_invalid_field_declarations(options):
    with pytest.raises((TypeError, ValueError)):
        SkimmingContext(4, **options)


@pytest.mark.parametrize("names", [("same", "same"), ("",), (1,), "cost"])
def test_invalid_output_names(names):
    with pytest.raises((TypeError, ValueError)):
        SkimmingOutputs(1, 4, names)


def test_fixed_dimensions_and_reinitialization():
    inputs = SkimmingContext(4, cost_name="objective")
    output = inputs.make_outputs(4)
    with pytest.raises(RuntimeError, match="reinitialized"):
        inputs.__init__(4, turn_cost_name="replacement")
    with pytest.raises(RuntimeError, match="reinitiali[sz]ed"):
        output.__init__(2, 4, ("replacement",))
    with pytest.raises(AttributeError):
        inputs.field_names = ("replacement",)
    with pytest.raises(AttributeError):
        output.origin_count = 2
    assert inputs.field_names == output.field_names == ("objective",)
    assert output.skims.shape == (1, 1, 4)
    with pytest.raises(ValueError, match="link_count"):
        SkimmingContext(-1)
    for origins, destinations in ((-1, 4), (1, -1)):
        with pytest.raises(ValueError, match="nonnegative"):
            inputs.make_outputs(destinations, origin_count=origins)


@pytest.mark.parametrize(
    "bad", ["row", "negative_row", "destinations", "link_count", "names", "order", "states", "width", "missing"]
)
def test_dimension_validation_precedes_writes(bad):
    context = history_context()
    inputs = SkimmingContext(4, link_fields={"distance": np.ones(4)}, cost_name="objective")
    results = search(context, 0)
    output = inputs.make_outputs(5 if bad == "destinations" else 4)
    scratch = workspace_for(context, inputs)
    if bad in ("names", "order"):
        output = SkimmingOutputs(1, 4, ("wrong", "objective") if bad == "names" else inputs.field_names[::-1])
    elif bad == "link_count":
        inputs = SkimmingContext(5, link_fields={"distance": np.ones(5)}, cost_name="objective")
    elif bad == "states":
        scratch = SkimmingWorkspace(context.state_count + 1, 1)
    elif bad == "width":
        scratch = SkimmingWorkspace(context.state_count, 2)
    elif bad == "missing":
        scratch = None
    before_output = output.skims.copy()
    before_scratch = None if scratch is None else scratch.state_skims.copy()
    row = {"row": 1, "negative_row": -1}.get(bad, 0)
    with pytest.raises(ValueError):
        skimming(results, inputs, scratch, output, origin_row=row)
    np.testing.assert_array_equal(output.skims, before_output)
    if scratch is not None:
        np.testing.assert_array_equal(scratch.state_skims, before_scratch)


def test_output_and_retained_views_do_not_retain_inputs():
    context = history_context()
    field = np.ones(4)
    field_ref = weakref.ref(field)
    inputs = SkimmingContext(4, link_fields={"distance": field}, cost_name="objective")
    output = inputs.make_outputs(4, origin_count=2)
    scratch = workspace_for(context, inputs)
    results = search(context, 0)
    del field
    gc.collect()
    assert field_ref() is not None
    skimming(results, inputs, scratch, output)
    matrices, state_sums = output.matrices, scratch.state_skims
    saved = output.skims.copy()
    del inputs, context, results, scratch
    gc.collect()
    assert field_ref() is None
    np.testing.assert_array_equal(output.skims, saved)
    del output
    gc.collect()
    assert not matrices["distance"].flags.c_contiguous
    np.testing.assert_array_equal(matrices["distance"], saved[:, 0, :])
    assert np.isfinite(state_sums).any()


@pytest.mark.parametrize("labels_only", [False, True])
def test_shared_inputs_and_disjoint_output_rows(labels_only):
    context = history_context(0.5)
    inputs = SkimmingContext(
        4,
        link_fields=None if labels_only else {"distance": np.ones(4)},
        cost_name="objective",
        turn_cost_name="penalty",
    )
    outputs = inputs.make_outputs(4, origin_count=4)
    results = [search(context, origin) for origin in range(4)]
    scratch = [workspace_for(context, inputs) for _ in results]

    def run(origin):
        return skimming(results[origin], inputs, scratch[origin], outputs, origin_row=origin)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(3):
            assert all(out is outputs for out in pool.map(run, range(4)))
    for origin in range(4):
        np.testing.assert_array_equal(
            outputs.skims[origin],
            path_walk_skims(results[origin], inputs, 4, cost_name="objective", turn_cost_name="penalty"),
        )


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("label", ["cost_name", "turn_cost_name"])
def test_label_only_assignment_searches_zero_demand_and_resets_skipped_rows(turn, label):
    context = history_context(0.5, turn=turn)
    inputs = SkimmingContext(4, **{label: "label"})
    demand = np.zeros((4, 4, 1))
    prepared = PreparedAoN(context, demand, skimming=inputs, cores=3)
    np.testing.assert_array_equal(prepared.origins, np.arange(4))
    np.testing.assert_array_equal(prepared.destination_counts, [4])
    output = prepared.run(prepared.make_outputs())
    retained = output.skimming.skims
    assert np.all(np.diag(retained[:, 0, :]) == 0)
    assert np.isfinite(retained[0, 0, 3])
    subset = PreparedAoN(context, demand, skimming=inputs, origins=[2])
    subset.run(output)
    assert np.all(np.isinf(retained[[0, 1, 3]]))
    assert retained[2, 0, 2] == 0
    component = output.skimming
    assert isinstance(component, SkimmingOutputs)
    assert np.shares_memory(retained, component.skims)
    del output, prepared, subset
    component.reset()
    skimming(search(context, 0), inputs, None, component, origin_row=3)
    assert component.skims[3, 0, 0] == 0


def test_assignment_rejects_mismatched_names_before_reset():
    context = history_context()
    demand = np.ones((4, 4, 1))
    first = PreparedAoN(context, demand, skimming=SkimmingContext(4, cost_name="first"))
    second = PreparedAoN(context, demand, skimming=SkimmingContext(4, cost_name="second"))
    out = first.run(first.make_outputs())
    saved_skims, saved_loads = out.skimming.skims.copy(), out.loading.link_loads.copy()
    with pytest.raises(ValueError, match="names"):
        second.run(out)
    np.testing.assert_array_equal(out.skimming.skims, saved_skims)
    np.testing.assert_array_equal(out.loading.link_loads, saved_loads)
    with pytest.raises(ValueError, match="link_count"):
        PreparedAoN(context, demand, skimming=SkimmingContext(5, cost_name="bad"))


@pytest.mark.parametrize("turn", [False, True])
def test_weighted_turn_costs_are_independent_of_skimming(turn):
    context = history_context(0.5, turn=turn)
    demand = np.array([[np.nan, np.inf], [2.0, -1.0], [4.0, 3.0], [5.0, 6.0]])
    query = LoadingQuery(demand)
    results = allocate_results(context)
    assert sum_weighted_turn_costs(results, query) == 0
    search(context, 0, results=results)
    expected = sum(demand[node, cls] * results.path_turn_cost_to(node) for node in range(1, 4) for cls in range(2))
    labels = results.turn_costs.copy()
    assert sum_weighted_turn_costs(results, query) == expected
    np.testing.assert_array_equal(results.turn_costs, labels)
    demand[1:] *= 2
    assert sum_weighted_turn_costs(results, query) == 2 * expected
    search(context, 0, targets=[1], results=results)
    assert sum_weighted_turn_costs(results, query) == 0
    assert sum_weighted_turn_costs(results, LoadingQuery(np.empty((0, 2)))) == 0
    assert sum_weighted_turn_costs(results, LoadingQuery(np.empty((4, 0)))) == 0
    demand[1] = np.inf
    assert np.isnan(sum_weighted_turn_costs(results, query))
    with pytest.raises(ValueError, match="destination_count"):
        sum_weighted_turn_costs(results, LoadingQuery(np.ones((5, 2))))
