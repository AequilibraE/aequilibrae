"""Exercise the existing driver as a consumer of the new routing boundary.

Driver, workspace and output composition will be revised separately. No legacy
Graph adapter or production assignment integration is part of these tests.
"""

import gc
import weakref

import numpy as np
import pytest

from aequilibrae.paths.cython.aon_context import PreparedAoN
from .routing_helpers import history_context, make_context, path_walk_outputs


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cores", [1, 3])
def test_selected_full_paths_and_output_rotation(turn, cores):
    context = history_context(turn=turn)
    demand = np.ones((4, 4, 2))
    selections = {"screenline": [0, 3, 3], "other": [2], "empty": []}
    fields = [np.ones(context.link_count)]
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=cores,
                           selected_links=selections, skim_fields=fields, skim_penalties=[True])
    previous, current = prepared.make_outputs(), prepared.make_outputs()
    retained = previous.link_loads
    for iteration in range(4):
        costs = np.array([1., 1. + iteration * 5, 1., 1.])
        prepared.update_costs(costs)
        expected = path_walk_outputs(context.with_costs(costs), demand, fields, [True], list(selections.values()))
        saved = previous.link_loads.copy()
        assert prepared.run(current) is current
        np.testing.assert_array_equal(previous.link_loads, saved)
        np.testing.assert_allclose(current.link_loads, expected[0])
        np.testing.assert_allclose(current.skims, expected[1])
        assert current.total_turn_penalty == expected[2]
        np.testing.assert_allclose(current.select_link_loads, expected[3])
        np.testing.assert_allclose(current.select_link_od, expected[4])
        previous, current = current, previous
    assert np.shares_memory(retained, previous.link_loads) or np.shares_memory(retained, current.link_loads)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("seed", [17, 29, 42])
def test_random_selected_loading_against_path_walks(turn, seed):
    rng = np.random.default_rng(seed)
    nodes = 8
    edges = [(a, b, float(rng.integers(0, 5))) for a in range(nodes) for b in range(nodes)
             for _ in range(2) if rng.random() < .1]
    turns = {}
    if turn:
        for incoming, (_, via, _) in enumerate(edges):
            for outgoing, (tail, _, _) in enumerate(edges):
                if via == tail and rng.random() < .25:
                    turns[incoming, outgoing] = np.inf if rng.random() < .2 else 5.
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=nodes))]
    context = make_context(fs, [b for _, b, _ in edges], [c for _, _, c in edges],
                           turns if turn else None, turn=turn)
    demand = rng.integers(-2, 5, size=(nodes, nodes, 2)).astype(np.float64)
    selected = {"first": list(range(0, context.link_count, 3)), "second": list(range(1, context.link_count, 2))}
    fields = [rng.random(context.link_count)]
    prepared = PreparedAoN(context, demand, costs=context.costs, cores=3, selected_links=selected, skim_fields=fields)
    out = prepared.run(prepared.make_outputs())
    expected = path_walk_outputs(context, demand, fields, selected_links=list(selected.values()))
    np.testing.assert_allclose(out.link_loads, expected[0])
    np.testing.assert_allclose(out.skims, expected[1])
    assert out.total_turn_penalty == expected[2]
    np.testing.assert_allclose(out.select_link_loads, expected[3])
    np.testing.assert_allclose(out.select_link_od, expected[4])


@pytest.mark.parametrize("turn", [False, True])
def test_independent_cost_bindings_and_no_input_retention_by_outputs(turn):
    context = history_context(turn=turn)
    demand = np.ones((4, 4, 1))
    costs = np.array([1., 20., 1., 1.] if turn else [20., 1., 1., 1.])
    costs_ref = weakref.ref(costs)
    a = PreparedAoN(context, demand, costs=context.costs)
    b = PreparedAoN(context, demand, costs=costs)
    first, second = a.run(a.make_outputs()), b.run(b.make_outputs())
    assert not np.array_equal(first.link_loads, second.link_loads)
    assert np.shares_memory(b.costs, costs)
    del costs, b
    gc.collect()
    assert costs_ref() is None
    view = second.link_loads
    snapshot = view.copy()
    del second
    gc.collect()
    np.testing.assert_array_equal(view, snapshot)


@pytest.mark.parametrize("turn", [False, True])
def test_blocking_uses_context_without_modifying_shared_heads(turn):
    context = make_context([0, 2, 3, 3], [1, 2, 2], [1, 10, 1], turn=turn)
    heads = context.heads.copy()
    demand = np.zeros((3, 3, 1))
    demand[0, 2, 0] = 1
    prepared = PreparedAoN(context, demand, costs=context.costs, block_centroids=True, cores=3)
    out = prepared.run(prepared.make_outputs())
    np.testing.assert_array_equal(out.link_loads[:, 0], [0, 1, 0])
    np.testing.assert_array_equal(context.heads, heads)
    assert context.blocked_centroid_count == 0


def test_repeated_runs_do_not_rebuild_inputs(monkeypatch):
    context = history_context()
    demand = np.ones((4, 4, 1))
    prepared = PreparedAoN(context, demand, costs=context.costs, skim_fields=[np.ones(4)],
                           selected_links={"set": [0, 3]}, cores=3)
    out = prepared.make_outputs()
    def unexpected(*args, **kwargs):
        raise AssertionError("setup should not run inside an iteration")
    for name in ("borrow_input", "copy_input", "make_destination_masks", "make_select_link_masks"):
        monkeypatch.setattr(f"aequilibrae.paths.cython.aon_context.{name}", unexpected)
    for _ in range(3):
        prepared.run(out)


@pytest.mark.parametrize("turn", [False, True])
def test_no_active_origins_resets_outputs(turn):
    context = make_context([0, 0, 0], [], [], turn=turn)
    prepared = PreparedAoN(context, np.zeros((2, 2, 1)), costs=context.costs, selected_links={"empty": []})
    out = prepared.run(prepared.make_outputs())
    assert out.total_turn_penalty == 0
    assert np.all(out.select_link_od == 0)
    assert out.select_link_loads.shape == (1, 0, 1)


def test_large_input_buffers_must_already_be_packed():
    context = history_context()
    demand = np.ones((4, 4, 1))
    for bad in (demand.astype(np.float32), np.ones((4, 8, 1))[:, ::2], demand.tolist()):
        with pytest.raises((ValueError, TypeError)):
            PreparedAoN(context, bad, costs=context.costs)
    with pytest.raises(ValueError, match="C-contiguous"):
        PreparedAoN(context, demand, costs=context.costs, skim_fields=[np.ones(8)[::2]])
