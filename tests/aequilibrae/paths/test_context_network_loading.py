"""Loading kernels still consume the same state tree through a const view."""

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from .routing_helpers import history_context, make_context, path_walk_outputs


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("penalty", [0.5, 10., np.inf])
@pytest.mark.parametrize("cores", [1, 3])
def test_loading_and_turn_totals_against_path_walks(turn, penalty, cores):
    context = history_context(penalty, turn=turn)
    demand = np.arange(48, dtype=np.float64).reshape(4, 4, 3) - 10
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=cores)
    out = prepared.make_outputs()
    expected, _, total, _, _ = path_walk_outputs(context, demand)
    for _ in range(3):
        assert prepared.run(out) is out
        np.testing.assert_allclose(out.link_loads, expected)
        assert out.total_turn_penalty == total


def test_borrowed_demand_values_are_used_without_copying():
    context = history_context()
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs)
    out = prepared.run(prepared.make_outputs())
    first = out.link_loads.copy()
    assert demand.flags.writeable
    # Keep the prepared target set unchanged; only demand magnitudes change here.
    demand *= 2
    prepared.run(out)
    np.testing.assert_array_equal(out.link_loads, first * 2)


@pytest.mark.parametrize("turn", [False, True])
def test_edgeless_network_and_intrazonal_demand(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    demand = np.ones((2, 2, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=3)
    out = prepared.run(prepared.make_outputs())
    assert out.link_loads.shape == (0, 1)
    assert out.total_turn_penalty == 0


@pytest.mark.parametrize("turn", [False, True])
def test_demand_classes_do_not_cancel_target_selection(turn):
    context = history_context(turn=turn)
    demand = np.zeros((4, 4, 2))
    demand[0, 3] = [1, -1]
    prepared = PreparedAoN(context, demand, costs=context.costs)
    out = prepared.run(prepared.make_outputs())
    expected = path_walk_outputs(context, demand)[0]
    assert np.any(expected)
    np.testing.assert_array_equal(out.link_loads, expected)
