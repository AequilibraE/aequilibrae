"""Output blends preserve storage, weight conventions and whole-group validation."""

import logging

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.context import SelectLinkContext, SkimmingContext
from aequilibrae.paths.cython.network_loading import network_loading
from aequilibrae.paths.cython.outputs import (
    AoNOutputs,
    LoadingOutputs,
    SelectLinkLoadingOutputs,
    SelectLinkODOutputs,
    SelectLinkOutputs,
    SkimmingOutputs,
)
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.select_link_loading import select_link_loading
from aequilibrae.paths.cython.skimming import skimming
from aequilibrae.paths.cython.workspaces import LoadingWorkspace, SelectLinkWorkspace

from .routing_helpers import make_context, search


SKIMS = ("cost", "turns")
SETS = ("first", "last")
KINDS = ("loading", "skimming", "selected_loading", "selected_od", "select_link", "aon")
SOURCE_COUNTS = {"copy_from": 1, "blend_cfw": 2, "blend_bfw": 3, "blend_result": 2}
WEIGHTS = {
    "copy_from": (1.0,),
    "blend_cfw": (0.7, 0.3),
    "blend_bfw": (0.2, 0.3, 0.5),
    "blend_result": (0.3, 0.7),
}


def loaded_output(values):
    """Populate link loads through routing, without bypassing read-only views."""
    values = np.asarray(values, dtype=np.float64)
    links, classes = values.shape
    output = LoadingOutputs(links, classes)
    if not links or not classes:
        return output

    # Each destination has one direct link from the origin, so its demand is
    # exactly the load on that link.
    context = make_context([0] + [links] * (links + 1), list(range(1, links + 1)), np.ones(links))
    demand = np.vstack((np.zeros((1, classes)), values))
    network_loading(search(context, 0), LoadingQuery(demand), LoadingWorkspace(context.state_count, classes), output)
    np.testing.assert_array_equal(output.link_loads, values)
    return output


def make_output(**options):
    return AoNOutputs(4, 4, 3, **{"skim_names": SKIMS, "select_link_names": SETS, **options})


def component(output, kind):
    if kind == "aon":
        return output
    if kind == "selected_loading":
        return output.select_link.loading
    if kind == "selected_od":
        return output.select_link.od
    return getattr(output, kind)


def arrays(output):
    if isinstance(output, AoNOutputs):
        result = arrays(output.loading)
        if output.skimming is not None:
            result += arrays(output.skimming)
        if output.select_link is not None:
            result += arrays(output.select_link)
        return result
    if isinstance(output, SelectLinkOutputs):
        return (arrays(output.loading) if output.loading is not None else []) + (
            arrays(output.od) if output.od is not None else []
        )
    if isinstance(output, SkimmingOutputs):
        return [output.skims]
    if isinstance(output, SelectLinkODOutputs):
        return [output.demand]
    return [output.link_loads]


def snapshot(output):
    return [value.copy() for value in arrays(output)], getattr(output, "turn_cost_total", None)


def assert_snapshot(output, saved):
    for actual, expected in zip(arrays(output), saved[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    if isinstance(output, AoNOutputs):
        assert output.turn_cost_total == saved[1]


def apply(output, method, sources, **options):
    sources = sources[: SOURCE_COUNTS[method]]
    if method == "copy_from":
        return output.copy_from(*sources)
    if method == "blend_bfw":
        weight = options.pop("weight", WEIGHTS[method])
    else:
        weight = options.pop("weight", 0.3)
    return getattr(output, method)(*sources, weight, **options)


@pytest.fixture
def sources():
    """Populate through public routing operations, never bypassing read-only views."""
    result = []
    for scale in (1.0, 2.0, 5.0):
        # The directed cycle reaches every destination. Distinct costs, demand
        # columns and selections expose axis swaps as well as reversed weights.
        context = make_context(
            [0, 1, 2, 3, 4],
            [1, 2, 3, 0],
            np.arange(1, 5, dtype=float) * scale,
            turns={(0, 1): 2.0 * scale, (1, 2): 3.0 * scale},
            turn=True,
        )
        demand = np.arange(1, 49, dtype=float).reshape(4, 4, 3) * scale
        prepared = PreparedAoN(
            context,
            demand,
            skimming=SkimmingContext(4, cost_name=SKIMS[0], turn_cost_name=SKIMS[1]),
            selected_links=SelectLinkContext(4, {SETS[0]: [0], SETS[1]: [2, 3]}),
        )
        result.append(prepared.run(prepared.make_outputs()))
    return result


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "method,alias",
    [(method, alias) for method, count in SOURCE_COUNTS.items() for alias in range(-1, count)],
)
@pytest.mark.parametrize("cores,threshold", [(1, -1), (2, 0), (2, 10000)])
def test_blends_and_copies_preserve_storage_and_allow_any_source_as_destination(
    sources, kind, method, alias, cores, threshold
):
    inputs = [component(source, kind) for source in sources]
    saved = [snapshot(source) for source in inputs]
    output = component(make_output(), kind) if alias == -1 else inputs[alias]
    retained = arrays(output)
    addresses = [value.ctypes.data for value in retained]
    weights = WEIGHTS[method]
    expected = [
        sum(weight * saved[index][0][field] for index, weight in enumerate(weights)) for field in range(len(retained))
    ]

    assert apply(output, method, inputs, cores=cores, threading_threshold=threshold) is output

    for actual, view, values, address in zip(arrays(output), retained, expected, addresses, strict=True):
        np.testing.assert_allclose(actual, values)
        np.testing.assert_allclose(view, values)
        assert actual.ctypes.data == address
        assert not actual.flags.writeable
    if kind == "aon":
        assert output.turn_cost_total == pytest.approx(sum(w * saved[i][1] for i, w in enumerate(weights)))
    for index, source in enumerate(inputs):
        if index != alias:
            assert_snapshot(source, saved[index])

    # Views retained before the copy/blend must also observe subsequent resets.
    output.reset()
    empty = component(make_output(), kind)
    assert_snapshot(output, snapshot(empty))
    for view, expected in zip(retained, arrays(empty), strict=True):
        np.testing.assert_array_equal(view, expected)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ("blend_cfw", "blend_result", "blend_bfw"))
@pytest.mark.parametrize("endpoint", (0, 1))
def test_weight_endpoints_with_finite_values(sources, kind, method, endpoint):
    inputs = [component(source, kind) for source in sources]
    output = component(make_output(), kind)
    weights = np.eye(3)[endpoint] if method == "blend_bfw" else endpoint
    apply(output, method, inputs, weight=weights)
    index = endpoint if method == "blend_bfw" else (endpoint if method == "blend_cfw" else 1 - endpoint)
    assert_snapshot(output, snapshot(inputs[index]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ("blend_cfw", "blend_result", "blend_bfw"))
@pytest.mark.parametrize("weight", (-0.1, 1.1, np.nan, np.inf, -np.inf))
def test_invalid_weights_leave_destination_unchanged(sources, kind, method, weight):
    inputs = [component(source, kind) for source in sources]
    output = inputs[-1]
    saved = snapshot(output)
    if method == "blend_bfw":
        weight = (weight, 0.0, 1.0)
    with pytest.raises(ValueError, match="weight"):
        apply(output, method, inputs, weight=weight)
    assert_snapshot(output, saved)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "weight",
    ((), (0.5, 0.5), (0.25, 0.25, 0.25, 0.25), [[0.2, 0.3, 0.5]], (0.2, 0.3, 0.4), (0.0, 0.0, 0.0)),
)
def test_invalid_bfw_shape_or_sum_leaves_destination_unchanged(sources, kind, weight):
    inputs = [component(source, kind) for source in sources]
    output = inputs[-1]
    saved = snapshot(output)
    with pytest.raises(ValueError, match="weight"):
        apply(output, "blend_bfw", inputs, weight=weight)
    assert_snapshot(output, saved)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ("blend_cfw", "blend_result", "blend_bfw"))
@pytest.mark.parametrize("cores", (0, -1))
def test_invalid_core_count_leaves_destination_unchanged(sources, kind, method, cores):
    inputs = [component(source, kind) for source in sources]
    saved = snapshot(inputs[-1])
    with pytest.raises(ValueError, match="cores"):
        apply(inputs[-1], method, inputs, cores=cores)
    assert_snapshot(inputs[-1], saved)


MISMATCHES = [
    ("loading", lambda: LoadingOutputs(5, 3)),
    ("loading", lambda: LoadingOutputs(4, 2)),
    ("skimming", lambda: SkimmingOutputs(3, 4, SKIMS)),
    ("skimming", lambda: SkimmingOutputs(4, 3, SKIMS)),
    ("skimming", lambda: SkimmingOutputs(4, 4, SKIMS[::-1])),
    ("skimming", lambda: SkimmingOutputs(4, 4, ("other", "turns"))),
    ("skimming", lambda: SkimmingOutputs(4, 4, ("cost",))),
    ("selected_loading", lambda: SelectLinkLoadingOutputs(3, 3, SETS)),
    ("selected_loading", lambda: SelectLinkLoadingOutputs(4, 2, SETS)),
    ("selected_loading", lambda: SelectLinkLoadingOutputs(4, 3, SETS[::-1])),
    ("selected_od", lambda: SelectLinkODOutputs(3, 4, 3, SETS)),
    ("selected_od", lambda: SelectLinkODOutputs(4, 3, 3, SETS)),
    ("selected_od", lambda: SelectLinkODOutputs(4, 4, 2, SETS)),
    ("selected_od", lambda: SelectLinkODOutputs(4, 4, 3, SETS[::-1])),
    # Loading matches, but the later OD component does not. A failed group
    # operation must not copy/blend loading before discovering this mismatch.
    ("select_link", lambda: SelectLinkOutputs(4, 3, 3, SETS, origin_count=4)),
    ("select_link", lambda: SelectLinkOutputs(4, 4, 3, SETS, origin_count=3)),
    ("select_link", lambda: SelectLinkOutputs(4, 4, 3, SETS, origin_count=4, od=False)),
    ("select_link", lambda: SelectLinkOutputs(4, 4, 3, SETS, origin_count=4, link_loads=False)),
    ("aon", lambda: make_output(skim_names=SKIMS[::-1])),
    ("aon", lambda: make_output(skim_names=())),
    ("aon", lambda: make_output(select_link_names=SETS[::-1])),
    ("aon", lambda: make_output(select_link_names=())),
    ("aon", lambda: make_output(select_link_od=False)),
    ("aon", lambda: make_output(select_link_loads=False)),
    ("aon", lambda: AoNOutputs(4, 3, 3, skim_names=SKIMS, select_link_names=SETS)),
]


@pytest.mark.parametrize("kind,bad_source", MISMATCHES)
@pytest.mark.parametrize(
    "method,position",
    [(method, position) for method, count in SOURCE_COUNTS.items() for position in range(count)],
)
def test_every_source_is_validated_before_any_component_changes(sources, kind, bad_source, method, position):
    inputs = [component(source, kind) for source in sources]
    output = inputs[-1]
    saved = snapshot(output)
    inputs[position] = bad_source()
    with pytest.raises(ValueError, match="must match"):
        apply(output, method, inputs)
    assert_snapshot(output, saved)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", SOURCE_COUNTS)
@pytest.mark.parametrize("bad_source", (None, object()))
def test_wrong_source_types_do_not_change_destination(sources, kind, method, bad_source):
    inputs = [component(source, kind) for source in sources]
    output = inputs[-1]
    saved = snapshot(output)
    inputs[SOURCE_COUNTS[method] - 1] = bad_source
    with pytest.raises(TypeError):
        apply(output, method, inputs)
    assert_snapshot(output, saved)


@pytest.mark.parametrize("skims", (False, True))
@pytest.mark.parametrize("loading,od", ((False, False), (False, True), (True, False), (True, True)))
@pytest.mark.parametrize("method", SOURCE_COUNTS)
def test_matching_optional_components(sources, skims, loading, od, method):
    options = {"skim_names": SKIMS if skims else (), "select_link_loads": loading, "select_link_od": od}
    # Copy populated components individually to exercise every enabled layout.
    inputs = [make_output(**options) for _ in sources]
    for output, source in zip(inputs, sources, strict=True):
        output.loading.copy_from(source.loading)
        if skims:
            output.skimming.copy_from(source.skimming)
        if loading:
            output.select_link.loading.copy_from(source.select_link.loading)
        if od:
            output.select_link.od.copy_from(source.select_link.od)
    output = make_output(**options)
    apply(output, method, inputs)
    for index, actual in enumerate(arrays(output)):
        expected = sum(w * arrays(inputs[i])[index] for i, w in enumerate(WEIGHTS[method]))
        np.testing.assert_allclose(actual, expected)


EMPTY_FACTORIES = [
    lambda: LoadingOutputs(0, 0),
    lambda: LoadingOutputs(0, 2),
    lambda: LoadingOutputs(3, 0),
    lambda: SkimmingOutputs(0, 3, SKIMS),
    lambda: SkimmingOutputs(3, 0, SKIMS),
    lambda: SkimmingOutputs(3, 4, ()),
    lambda: SelectLinkLoadingOutputs(0, 3, SETS),
    lambda: SelectLinkLoadingOutputs(3, 0, SETS),
    lambda: SelectLinkLoadingOutputs(3, 4, ()),
    lambda: SelectLinkODOutputs(0, 4, 3, SETS),
    lambda: SelectLinkODOutputs(3, 0, 3, SETS),
    lambda: SelectLinkODOutputs(3, 4, 0, SETS),
    lambda: SelectLinkODOutputs(3, 4, 3, ()),
    lambda: SelectLinkOutputs(3, 4, 3, SETS, origin_count=2, link_loads=False, od=False),
    lambda: AoNOutputs(0, 0, 0, skim_names=SKIMS, select_link_names=SETS),
]


@pytest.mark.parametrize("factory", EMPTY_FACTORIES)
@pytest.mark.parametrize("method", SOURCE_COUNTS)
def test_empty_axes_and_empty_groups(factory, method):
    output = factory()
    inputs = [factory() for _ in range(3)]
    saved = snapshot(output)
    assert apply(output, method, inputs) is output
    assert_snapshot(output, saved)
    if method != "copy_from":
        with pytest.raises(ValueError):
            apply(output, method, inputs, weight=-1.0)


@pytest.mark.parametrize(
    "method,weight",
    [
        ("blend_cfw", 0.0),
        ("blend_cfw", 1.0),
        ("blend_result", 0.0),
        ("blend_result", 1.0),
        ("blend_bfw", (0.0, 0.3, 0.7)),
        ("blend_bfw", (0.3, 0.0, 0.7)),
        ("blend_bfw", (0.3, 0.7, 0.0)),
    ],
)
@pytest.mark.parametrize("grouped", (False, True))
def test_zero_weight_skims_warn_and_keep_existing_infinity_arithmetic(caplog, method, weight, grouped):
    # Unwritten skim entries start at infinity. A zero weight is deliberately
    # not a shortcut to a copy: the existing helpers yield NaN at those entries.
    outputs = [make_output() for _ in range(4)]
    if not grouped:
        outputs = [output.skimming for output in outputs]
    with caplog.at_level(logging.WARNING, logger="aequilibrae.paths.cython.outputs"):
        apply(outputs[0], method, outputs[1:], weight=weight)
    skims = outputs[0].skimming.skims if grouped else outputs[0].skims
    assert np.isnan(skims).all()
    warnings = [record for record in caplog.records if "zero weight blend" in record.message]
    assert len(warnings) == 1
    assert warnings[0].name == "aequilibrae.paths.cython.outputs"


@pytest.mark.parametrize("method", SOURCE_COUNTS)
def test_positive_weight_blends_and_copies_do_not_warn(caplog, method):
    inputs = [make_output() for _ in range(3)]
    output = make_output()
    with caplog.at_level(logging.WARNING):
        apply(output, method, inputs)
    assert np.isinf(output.skimming.skims).all()
    assert not caplog.records


@pytest.mark.parametrize("method", SOURCE_COUNTS)
@pytest.mark.parametrize("kind", ("skimming", "selected_od"))
def test_rectangular_outputs_and_retained_named_views(method, kind):
    context = make_context([0, 1, 2, 3, 4], [1, 2, 3, 0], np.ones(4))
    inputs = []
    for scale in (1.0, 2.0, 5.0):
        context.update_costs(np.full(4, scale))
        skim_context = SkimmingContext(4, cost_name="cost", turn_cost_name="turns")
        selections = SelectLinkContext(4, {"first": [0], "last": [2, 3]})
        # Neither output is square, and origins != sets. This catches mistakes
        # when the four OD axes are folded into a view for the 3D helpers.
        output = SkimmingOutputs(3, 4, SKIMS) if kind == "skimming" else SelectLinkODOutputs(3, 4, 5, SETS)
        for row, origin in enumerate((3, 1, 0)):
            results = search(context, origin)
            if kind == "skimming":
                skimming(results, skim_context, None, output, origin_row=row)
            else:
                demand = np.arange(20, dtype=float).reshape(4, 5) * scale * (row + 1)
                select_link_loading(
                    results,
                    LoadingQuery(demand),
                    selections,
                    SelectLinkWorkspace(context.state_count),
                    None,
                    None,
                    output,
                    origin_row=row,
                )
        inputs.append(output)
    output = inputs[-1]
    retained = output.matrices
    saved = [arrays(source)[0].copy() for source in inputs]
    expected = sum(w * saved[i] for i, w in enumerate(WEIGHTS[method]))

    apply(output, method, inputs)

    np.testing.assert_allclose(arrays(output)[0], expected)
    for index, view in enumerate(retained.values()):
        np.testing.assert_allclose(view, expected[:, index])
        assert not view.flags.writeable
        assert np.shares_memory(view, arrays(output)[0])
    output.reset()
    for view in retained.values():
        assert np.isinf(view).all() if kind == "skimming" else not np.any(view)


def test_project_to_network_uses_crosswalk_for_each_network_link():
    output = LoadingOutputs(4, 2)
    compact = loaded_output([[2, 4], [7, 1], [3, 9]])
    crosswalk = np.array([2, 0, 2, 1], dtype=np.int64)

    output.copy_from_compact(compact, crosswalk)

    assert output.link_loads.shape == (4, 2)
    np.testing.assert_array_equal(output.link_loads, compact.link_loads[[2, 0, 2, 1]])


def test_project_to_network_zeroes_removed_links_without_a_dummy_source_row():
    compact = loaded_output([[2, 4], [7, 1]])
    output = loaded_output([[99, 99], [99, 99], [99, 99]])
    output.copy_from_compact(compact, np.array([1, 2, 0], dtype=np.int64))
    np.testing.assert_array_equal(output.link_loads, [[7, 1], [0, 0], [2, 4]])


@pytest.mark.parametrize("classes", [0, 2])
def test_project_empty_compact_network(classes):
    compact = LoadingOutputs(0, classes)
    output = LoadingOutputs(3, classes)
    output.copy_from_compact(compact, np.zeros(3, dtype=np.int64))
    assert not np.any(output.link_loads)


@pytest.mark.parametrize("bad_index", [-1, 4])
def test_project_rejects_bad_index_before_any_writes(bad_index):
    output = loaded_output([[1], [2], [3]])
    compact = loaded_output([[9], [8], [7]])
    before = output.link_loads.copy()
    with pytest.raises(ValueError, match="invalid compact link"):
        output.copy_from_compact(compact, np.array([0, 1, bad_index], dtype=np.int64))
    np.testing.assert_array_equal(output.link_loads, before)


@pytest.mark.parametrize("crosswalk", ([[-1]], [3], [0.5]))
def test_project_to_network_rejects_invalid_crosswalk(crosswalk):
    output = LoadingOutputs(3, 1)
    compact = LoadingOutputs(3, 1)

    with pytest.raises((TypeError, ValueError)):
        output.copy_from_compact(compact, crosswalk)


def test_group_bfw_snapshots_weights_before_overwriting_their_source(sources):
    output = sources[0]
    # Construct a valid weight row through the public loading operation rather
    # than mutating the protected buffer. This row aliases the first component
    # written by the grouped operation.
    context = make_context([0, 1, 2, 3, 4], [1, 2, 3, 0], np.ones(4))
    demand = np.zeros((4, 3))
    demand[1] = [0.2, 0.3, 0.5]
    output.loading.reset()
    network_loading(search(context, 0), LoadingQuery(demand), LoadingWorkspace(context.state_count, 3), output.loading)
    weights = output.loading.link_loads[0]
    saved_weights = weights.copy()
    saved = [snapshot(source) for source in sources]

    output.blend_bfw(*sources, weights)

    for field, actual in enumerate(arrays(output)):
        expected = sum(w * saved[i][0][field] for i, w in enumerate(saved_weights))
        np.testing.assert_allclose(actual, expected)
    assert output.turn_cost_total == pytest.approx(sum(w * saved[i][1] for i, w in enumerate(saved_weights)))
