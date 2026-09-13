"""Keep skim-kernel checks while its input/output wrappers are being separated."""

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from .routing_helpers import history_context, make_context, path_walk_outputs


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("penalty", [0.5, 10., np.inf])
@pytest.mark.parametrize("width", [0, 1, 3, 37])
def test_skim_state_tree_against_path_sums(turn, penalty, width):
    context = history_context(penalty, turn=turn)
    fields = [np.arange(context.link_count, dtype=np.float64) + i for i in range(width)]
    penalties = [i % 2 == 0 for i in range(width)]
    demand = np.ones((4, 4, 2))
    prepared = PreparedAoN(context, demand, costs=context.costs,
                           skim_fields=fields, skim_penalties=penalties, cores=3)
    out = prepared.make_outputs()
    prepared.run(out)
    expected = path_walk_outputs(context, demand, fields, penalties)[1]
    if width:
        np.testing.assert_allclose(out.skims, expected)
    else:
        assert out.skims is None


def test_skim_fields_are_borrowed_without_changing_writeability():
    context = history_context()
    field = np.ones(context.link_count)
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, skim_fields=[field])
    out = prepared.make_outputs()
    prepared.run(out)
    assert out.skims[0, 3, 0] == 3
    assert field.flags.writeable
    field[:] = 2
    prepared.run(out)
    assert out.skims[0, 3, 0] == 6
    # Field meaning does not change when the routing objective is rebound.
    prepared.update_costs(np.array([1., 10., 1., 1.]))
    prepared.run(out)
    assert out.skims[0, 3, 0] == 4


@pytest.mark.parametrize("turn", [False, True])
def test_centroid_skims_use_intermediate_states(turn):
    # The two output nodes are centroids; both paths use node 2 outside the OD axes.
    context = make_context([0, 1, 2, 4], [2, 2, 0, 1], [1, 1, 1, 1], turn=turn)
    demand = np.ones((2, 2, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, skim_fields=[np.ones(4)])
    out = prepared.run(prepared.make_outputs())
    np.testing.assert_array_equal(out.skims[:, :, 0], [[0, 2], [2, 0]])
