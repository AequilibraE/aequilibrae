"""Loading blends preserve output ownership and assignment weight conventions."""

import numpy as np
import pytest
from aequilibrae.paths.cython.network_loading import network_loading
from aequilibrae.paths.cython.outputs import LoadingOutputs
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.workspaces import LoadingWorkspace

from .routing_helpers import make_context, search


def loaded_output(values):
    """Create an output through network loading"""
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


@pytest.fixture
def sources():
    return (
        loaded_output([[2, 4], [7, 1], [3, 9]]),
        loaded_output([[5, 1], [4, 8], [2, 6]]),
        loaded_output([[1, 8], [9, 5], [6, 2]]),
    )


def test_copy_from_replaces_loads_without_replacing_storage(sources):
    source, _, _ = sources
    output = loaded_output(np.full((3, 2), 100.0))
    retained = output.link_loads
    address = retained.ctypes.data
    source_before = source.link_loads.copy()

    assert output.copy_from(source) is output

    np.testing.assert_array_equal(retained, source_before)
    np.testing.assert_array_equal(source.link_loads, source_before)
    assert output.link_loads.ctypes.data == address
    assert not retained.flags.writeable

    output.reset()
    np.testing.assert_array_equal(retained, np.zeros((3, 2)))


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (3, 0)])
def test_copy_from_handles_empty_outputs(shape):
    output = LoadingOutputs(*shape)
    source = LoadingOutputs(*shape)

    assert output.copy_from(source) is output
    assert output.link_loads.shape == shape
    assert output.link_loads.size == 0


def test_project_to_network_uses_crosswalk_for_each_network_link():
    aon = LoadingOutputs(4, 2)
    compact = loaded_output([[2, 4], [7, 1], [3, 9]])
    crosswalk = np.array([2, 0, 2, 1], dtype=np.int64)

    aon.copy_from_compact(compact, crosswalk)

    assert aon.link_loads.shape == (4, 2)
    np.testing.assert_array_equal(
        aon.link_loads,
        compact.link_loads[[2, 0, 2, 1]],
    )


@pytest.mark.parametrize("crosswalk", ([[-1]], [3], [0.5]))
def test_project_to_network_rejects_invalid_crosswalk(crosswalk):
    aon = LoadingOutputs(3, 1)
    compact = LoadingOutputs(3, 1)

    with pytest.raises((TypeError, ValueError)):
        aon.copy_from_compact(compact, crosswalk)


def test_copy_from_rejects_mismatched_source_without_changing_destination(sources):
    output, _, _ = sources
    before = output.link_loads.copy()

    with pytest.raises(ValueError, match="dimensions must match"):
        output.copy_from(LoadingOutputs(4, 2))

    np.testing.assert_array_equal(output.link_loads, before)


@pytest.mark.parametrize("cores, threading_threshold", [(1, -1), (2, 0), (2, 10000)])
def test_blend_cfw_uses_the_conjugate_weight_for_the_previous_direction(sources, cores, threading_threshold):
    aon, previous_direction, _ = sources
    output = LoadingOutputs(3, 2)
    conjugate_weight = 0.3
    expected = conjugate_weight * previous_direction.link_loads + (1.0 - conjugate_weight) * aon.link_loads

    assert (
        output.blend_cfw(
            aon,
            previous_direction,
            conjugate_weight,
            cores=cores,
            threading_threshold=threading_threshold,
        )
        is output
    )

    np.testing.assert_allclose(output.link_loads, expected)


def test_blend_cfw_allows_in_place_direction_update(sources):
    aon, previous_direction, _ = sources
    conjugate_weight = 0.3
    expected = conjugate_weight * previous_direction.link_loads + (1.0 - conjugate_weight) * aon.link_loads

    previous_direction.blend_cfw(aon, previous_direction, conjugate_weight)

    np.testing.assert_allclose(previous_direction.link_loads, expected)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_blend_cfw_weight_endpoints(sources, weight):
    aon, previous_direction, _ = sources
    output = LoadingOutputs(3, 2)

    output.blend_cfw(aon, previous_direction, weight)

    expected = weight * previous_direction.link_loads + (1.0 - weight) * aon.link_loads
    np.testing.assert_array_equal(output.link_loads, expected)


@pytest.mark.parametrize("weight", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_blend_cfw_rejects_invalid_weight_without_changing_destination(sources, weight):
    aon, previous_direction, output = sources
    before = output.link_loads.copy()

    with pytest.raises(ValueError, match="blend weight"):
        output.blend_cfw(aon, previous_direction, weight)

    np.testing.assert_array_equal(output.link_loads, before)


@pytest.mark.parametrize("cores, threading_threshold", [(1, -1), (2, 0), (2, 10000)])
def test_blend_bfw_uses_all_three_directions(sources, cores, threading_threshold):
    aon, previous_direction, older_direction = sources
    output = LoadingOutputs(3, 2)
    weights = np.array([0.2, 0.3, 0.5])
    expected = (
        weights[0] * aon.link_loads
        + weights[1] * previous_direction.link_loads
        + weights[2] * older_direction.link_loads
    )

    assert (
        output.blend_bfw(
            aon,
            previous_direction,
            older_direction,
            weights,
            cores=cores,
            threading_threshold=threading_threshold,
        )
        is output
    )

    np.testing.assert_allclose(output.link_loads, expected)


@pytest.mark.parametrize(
    "weights",
    [
        (),
        (0.5, 0.5),
        (0.25, 0.25, 0.25, 0.25),
        [[0.2, 0.3, 0.5]],
        (np.nan, 0.0, 1.0),
        (-0.1, 0.5, 0.6),
        (1.1, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.2, 0.3, 0.6),
    ],
)
def test_blend_bfw_rejects_invalid_weights_without_changing_destination(sources, weights):
    aon, previous_direction, output = sources
    before = output.link_loads.copy()

    with pytest.raises(ValueError):
        output.blend_bfw(aon, previous_direction, output, weights)

    np.testing.assert_array_equal(output.link_loads, before)


@pytest.mark.parametrize("cores, threading_threshold", [(1, -1), (2, 0), (2, 10000)])
def test_blend_result_uses_the_stepsize_for_the_new_direction(sources, cores, threading_threshold):
    direction, previous_result, _ = sources
    output = LoadingOutputs(3, 2)
    stepsize = 0.4
    expected = stepsize * direction.link_loads + (1.0 - stepsize) * previous_result.link_loads

    assert (
        output.blend_result(
            direction,
            previous_result,
            stepsize,
            cores=cores,
            threading_threshold=threading_threshold,
        )
        is output
    )

    np.testing.assert_allclose(output.link_loads, expected)


def test_blend_result_can_update_the_previous_result_in_place(sources):
    direction, previous_result, _ = sources
    stepsize = 0.4
    expected = stepsize * direction.link_loads + (1.0 - stepsize) * previous_result.link_loads

    previous_result.blend_result(direction, previous_result, stepsize)

    np.testing.assert_allclose(previous_result.link_loads, expected)


@pytest.mark.parametrize("stepsize", [0.0, 1.0])
def test_blend_result_weight_endpoints(sources, stepsize):
    direction, previous_result, _ = sources
    output = LoadingOutputs(3, 2)

    output.blend_result(direction, previous_result, stepsize)

    expected = stepsize * direction.link_loads + (1.0 - stepsize) * previous_result.link_loads
    np.testing.assert_array_equal(output.link_loads, expected)


@pytest.mark.parametrize("stepsize", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_blend_result_rejects_invalid_stepsize_without_changing_destination(sources, stepsize):
    direction, previous_result, output = sources
    before = output.link_loads.copy()

    with pytest.raises(ValueError, match="blend weight"):
        output.blend_result(direction, previous_result, stepsize)

    np.testing.assert_array_equal(output.link_loads, before)


def _blend_cfw_with_cores(output, aon, previous_direction, older_direction, cores):
    output.blend_cfw(aon, previous_direction, 0.3, cores=cores)


def _blend_bfw_with_cores(output, aon, previous_direction, older_direction, cores):
    output.blend_bfw(aon, previous_direction, older_direction, (0.2, 0.3, 0.5), cores=cores)


def _blend_result_with_cores(output, aon, previous_direction, older_direction, cores):
    output.blend_result(previous_direction, older_direction, 0.4, cores=cores)


@pytest.mark.parametrize(
    "blend",
    [
        pytest.param(_blend_cfw_with_cores, id="cfw"),
        pytest.param(_blend_bfw_with_cores, id="bfw"),
        pytest.param(_blend_result_with_cores, id="result"),
    ],
)
@pytest.mark.parametrize("cores", [0, -1])
def test_blends_reject_nonpositive_cores_without_changing_destination(sources, blend, cores):
    aon, previous_direction, older_direction = sources
    output = loaded_output(np.full((3, 2), 100.0))
    before = output.link_loads.copy()

    with pytest.raises(ValueError, match="cores must be positive"):
        blend(output, aon, previous_direction, older_direction, cores)

    np.testing.assert_array_equal(output.link_loads, before)


def test_blends_handle_empty_outputs():
    aon = LoadingOutputs(0, 2)
    previous_direction = LoadingOutputs(0, 2)
    older_direction = LoadingOutputs(0, 2)
    output = LoadingOutputs(0, 2)

    output.blend_cfw(aon, previous_direction, 0.3)
    output.blend_bfw(aon, previous_direction, older_direction, (0.2, 0.3, 0.5))
    output.blend_result(previous_direction, aon, 0.4)

    assert output.link_loads.shape == (0, 2)
    assert output.link_loads.size == 0
