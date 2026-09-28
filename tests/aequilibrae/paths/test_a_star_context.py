"""A* state trees, explicit heuristics and PathResults integration."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import PathResults, available_heaps, estimate_heuristic_scale
from aequilibrae.paths.cython.a_star import (
    EuclideanContext, HaversineContext, a_star, estimate_context_scale,
)
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.results import path_results as path_results_module

from .routing_helpers import allocate_results, assert_state_tree, history_context, make_context, search
from .test_assignment_integration import diamond


def heuristic_for(context, name, x=None, y=None, scale=None):
    x = np.arange(context.node_count, dtype=float) if x is None else np.asarray(x, dtype=float)
    y = np.zeros(context.node_count) if y is None else np.asarray(y, dtype=float)
    cls = EuclideanContext if name == "euclidean" else HaversineContext
    if name == "haversine":
        x, y = x * 0.01 + 150, y * 0.01 - 30
    raw = cls(x, y, 1.0)
    return cls(x, y, estimate_context_scale(context, raw) if scale is None else scale)


def run(context, origin, destination, heuristic, heap="4ary", results=None):
    mask = np.zeros(context.node_count, dtype=bool)
    mask[destination] = True
    results = allocate_results(context) if results is None else results
    return a_star(context, SearchQuery(context.node_count, origin, mask), destination, heuristic, results, heap)


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("name", ["euclidean", "haversine"])
@pytest.mark.parametrize("penalty", [0.5, 10, np.inf])
def test_arrival_history_and_partial_labels(heap, name, penalty):
    context = history_context(penalty)
    heuristic = heuristic_for(context, name)
    results = run(context, 0, 3, heuristic, heap)
    reference = search(context, 0, 3)
    np.testing.assert_array_equal(results.path_links_to(3), reference.path_links_to(3))
    assert results.path_cost_to(3) == reference.path_cost_to(3)
    assert results.path_turn_cost_to(3) == reference.path_turn_cost_to(3)
    assert results.target_count == results.reached_target_count == 1
    assert not results.exhausted
    assert_state_tree(context, results)
    # Reusing the results clears labels, and starting at the via node pays no turn.
    run(context, 1, 3, heuristic, heap, results)
    assert results.path_cost_to(3) == 1
    assert results.path_turn_cost_to(3) == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("name", ["euclidean", "haversine"])
@pytest.mark.parametrize("turn", [False, True])
def test_heuristic_guides_search_and_zero_scale_matches_dijkstra(heap, name, turn):
    context = make_context([0, 2, 3, 3, 3], [1, 2, 3], [1, 1, 1], turn=turn)
    heuristic = heuristic_for(context, name, [0, 1, 0, 2], [0, 0, 1, 0])
    results = run(context, 0, 3, heuristic, heap)
    reference = search(context, 0, 3)
    assert results.path_cost_to(3) == 2
    assert results.settled_count < reference.settled_count
    assert not results.reachable_to(2)
    assert_state_tree(context, results)
    zero = heuristic_for(context, name, scale=0)
    run(context, 0, 3, zero, heap, results)
    assert results.settled_count == reference.settled_count
    assert_state_tree(context, results)


def test_parallel_links_keep_separate_turn_history():
    context = make_context([0, 2, 3, 3], [1, 1, 2], [1, 2, 1], {(0, 2): 10})
    results = run(context, 0, 2, heuristic_for(context, "euclidean"))
    np.testing.assert_array_equal(results.path_links_to(2), [1, 2])
    assert results.path_cost_to(2) == 3
    assert results.terminal_states[1] == 0
    assert_state_tree(context, results)


@pytest.mark.parametrize("reason", ["turn", "link", "overflow"])
def test_unreachable_nonfinite_transitions(reason):
    costs, penalty = [1, 1], np.inf
    if reason == "link":
        costs[1], penalty = np.inf, 0
    elif reason == "overflow":
        costs, penalty = [1e308, 1e308], 0
    context = make_context([0, 1, 2, 2], [1, 2], costs, {(0, 1): penalty})
    results = run(context, 0, 2, heuristic_for(context, "euclidean"))
    assert results.exhausted and not results.all_targets_reached
    assert results.path_cost_to(2) == np.inf
    assert_state_tree(context, results)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("name", ["euclidean", "haversine"])
def test_edgeless_intrazonal_and_unreachable(turn, name):
    context = make_context([0, 0, 0], [], [], turn=turn)
    heuristic = heuristic_for(context, name)
    for origin, destination in [(0, 0), (0, 1), (1, 1)]:
        results = run(context, origin, destination, heuristic)
        assert results.settled_count == 1
        assert results.exhausted == (origin != destination)
        assert results.path_cost_to(destination) == (0 if origin == destination else np.inf)
        assert_state_tree(context, results)


@pytest.mark.parametrize("turn", [False, True])
def test_centroid_blocking(turn):
    context = make_context([0, 1, 2, 3, 3], [1, 2, 3], [1, 1, 1], turn=turn, blocked_centroid_count=2)
    heuristic = heuristic_for(context, "euclidean")
    results = run(context, 0, 3, heuristic)
    assert results.exhausted and not results.reachable_to(3)
    assert results.reachable_to(1)
    assert_state_tree(context, results)
    results = run(context, 1, 3, heuristic)
    assert results.path_cost_to(3) == 2


@pytest.mark.parametrize("allow_uturns, override, reachable", [
    (True, None, True), (False, None, False), (False, 0, True), (False, 2, True), (True, np.inf, False),
])
@pytest.mark.parametrize("cost", [0, 1])
def test_uturns_explicit_overrides_and_zero_cost_cycles(allow_uturns, override, reachable, cost):
    turns = {(0, 2): np.inf}
    if override is not None:
        turns[1, 3] = override
    context = make_context([0, 1, 3, 4, 4], [1, 2, 3, 1], [cost] * 4, turns, allow_uturns=allow_uturns)
    results = run(context, 0, 3, heuristic_for(context, "euclidean"))
    assert results.reachable_to(3) == reachable
    if reachable:
        np.testing.assert_array_equal(results.path_links_to(3), [0, 1, 3, 2])
        assert results.path_cost_to(3) == 4 * cost + (override or 0)
    assert_state_tree(context, results)


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("name", ["euclidean", "haversine"])
def test_random_parallel_graph_matches_dijkstra(heap, turn, name):
    rng = np.random.default_rng(239)
    n = 9
    edges = [(a, b, rng.uniform(0.1, 10)) for a in range(n) for b in range(n)
             for _ in range(2) if rng.random() < 0.15]
    turns = {}
    for incoming, (_, via, _) in enumerate(edges):
        for outgoing, (tail, _, _) in enumerate(edges):
            if via == tail and rng.random() < 0.4:
                turns[incoming, outgoing] = rng.choice([0.0, 2.5, np.inf])
    fs = np.r_[0, np.cumsum(np.bincount([a for a, _, _ in edges], minlength=n))]
    context = make_context(fs, [b for _, b, _ in edges], [c for _, _, c in edges],
                           turns if turn else None, **({"allow_uturns": False} if turn else {}))
    heuristic = heuristic_for(context, name, rng.uniform(-5, 5, n), rng.uniform(-5, 5, n))
    results = allocate_results(context)
    for origin in range(n):
        reference = search(context, origin)
        for destination in range(n):
            run(context, origin, destination, heuristic, heap, results)
            assert results.path_cost_to(destination) == pytest.approx(reference.path_cost_to(destination))
            assert_state_tree(context, results)
            # Every published terminal, not only the destination, must be optimal.
            for node in range(n):
                if results.reachable_to(node):
                    assert results.path_cost_to(node) == pytest.approx(reference.path_cost_to(node))


def coordinates_for(graph):
    coordinates = pd.DataFrame({"x": [0., 1., 0., 2.], "y": [0., 0., 1., 0.]}, index=[10, 20, 30, 40])
    graph.lonlat_index = coordinates.rename(columns={"x": "lon", "y": "lat"}) * 0.01
    return coordinates.iloc[::-1]


@pytest.mark.parametrize("heap", available_heaps())
@pytest.mark.parametrize("name", ["euclidean", "haversine"])
@pytest.mark.parametrize("turn", [False, True])
def test_path_results_snapshot_skims_and_update_trace(heap, name, turn, monkeypatch):
    graph = diamond(turn)
    coordinates = coordinates_for(graph)
    scale = estimate_heuristic_scale(graph, coordinates, heuristic=name)
    result = PathResults(graph, 10, 20, coordinates=coordinates, a_star=True,
                         heuristic=name, heuristic_scale=scale, heap=heap)
    assert result.a_star and result.early_exit
    assert result.get_heuristics() == ["euclidean", "haversine"]
    assert result.milepost[-1] == 1
    assert_state_tree(result.context, result.search_results)
    # Mutating all sources must not affect a later search of the snapshot.
    coordinates.loc[:, :] = np.nan
    graph.lonlat_index.loc[:, :] = np.nan
    graph.cost[:] = 1000
    called = []
    original = path_results_module.run_a_star

    def record(context, query, destination, heuristic, results, heap):
        called.append((heap, heuristic.scale, type(heuristic)))
        return original(context, query, destination, heuristic, results, heap)

    monkeypatch.setattr(path_results_module, "run_a_star", record)
    result.set_heap("std" if heap != "std" else "4ary")
    result.set_heuristic("haversine" if name == "euclidean" else "euclidean")
    result.update_trace(40)
    assert called == [(heap, scale, EuclideanContext if name == "euclidean" else HaversineContext)]
    assert result.milepost[-1] == (2.5 if turn else 2)
    np.testing.assert_array_equal(result.path, [71, 12])
    np.testing.assert_array_equal(result.path_nodes, [10, 20, 40])
    target = int(np.flatnonzero(result.node_ids == 40)[0])
    assert result.skims.matrices["time"][0, target] == result.milepost[-1]
    assert result.skims.matrices["distance"][0, target] == 7
    assert_state_tree(result.context, result.search_results)
    result.update_trace(10)
    assert len(called) == 1
    np.testing.assert_array_equal(result.milepost, [0])
    result.reset()
    assert not result.a_star and not result.early_exit
    assert np.all(np.isinf(result.distances))


@pytest.mark.parametrize("scale", [None, -1, np.inf, np.nan])
def test_requires_valid_explicit_scale(scale):
    graph = diamond()
    with pytest.raises(ValueError, match="heuristic_scale"):
        PathResults(graph, 10, 40, a_star=True, coordinates=coordinates_for(graph), heuristic_scale=scale)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "columns"])
def test_coordinate_validation_is_deferred_for_dijkstra(problem):
    graph = diamond()
    coordinates = coordinates_for(graph)
    if problem == "missing":
        coordinates = coordinates.drop(20)
    elif problem == "duplicate":
        coordinates = pd.concat([coordinates, coordinates])
    elif problem == "nonfinite":
        coordinates.loc[20, "x"] = np.nan
    else:
        coordinates = coordinates.rename(columns={"x": "lon"})
    result = PathResults(graph, 10, 40, coordinates=coordinates)
    with pytest.raises(ValueError):
        result.compute_path(10, 40, a_star=True, heuristic_scale=0)
    assert result.milepost[-1] == 2


def test_missing_coordinates_and_heuristic_selection():
    graph = diamond()
    result = PathResults(graph, 10, 40)
    with pytest.raises(ValueError, match="coordinate columns"):
        result.compute_path(10, 40, a_star=True, heuristic_scale=1)
    with pytest.raises(ValueError, match="coordinate columns"):
        result.compute_path(10, 40, a_star=True, heuristic="haversine", heuristic_scale=1)
    with pytest.raises(ValueError, match="heuristic must be"):
        result.set_heuristic("equirectangular")
    coordinates = coordinates_for(graph)
    result.set_graph_data(graph, coordinates=coordinates)
    result.set_heuristic("haversine")
    result.compute_path(10, 40, a_star=True, heuristic_scale=0)
    assert isinstance(result._a_star_context, HaversineContext)


def test_haversine_constructor_selection_and_coordinate_snapshot():
    graph = diamond()
    coordinates_for(graph)
    result = PathResults(graph, 10, 40, heuristic="haversine")
    # Coordinates must be copied even when the initial search is Dijkstra.
    graph.lonlat_index.loc[:, :] = np.nan
    result.compute_path(10, 40, a_star=True, heuristic_scale=0)
    assert isinstance(result._a_star_context, HaversineContext)
    assert result.milepost[-1] == 2


def test_scale_helper_bounds_and_no_automatic_clamping():
    graph = diamond()
    coordinates = coordinates_for(graph)
    scale = estimate_heuristic_scale(graph, coordinates)
    assert scale == pytest.approx(2 / np.sqrt(5))
    # The helper does not change graph costs or supplied coordinates.
    np.testing.assert_array_equal(graph.cost, [1, 2, 1, 2])
    assert np.all(np.isfinite(coordinates))
    result = PathResults(graph, 10, 40, a_star=True, coordinates=coordinates, heuristic_scale=100)
    assert result._a_star_context.scale == 100
    graph.cost[0] = 0
    assert estimate_heuristic_scale(graph, coordinates) == 0
    graph.cost[:] = np.inf
    assert estimate_heuristic_scale(graph, coordinates) == 0
    graph.cost[:] = 1
    coordinates.loc[:, :] = 0
    assert estimate_heuristic_scale(graph, coordinates) == 0


def test_helper_is_not_called_by_path_results(monkeypatch):
    from aequilibrae.paths import path_heuristics
    graph = diamond()
    coordinates = coordinates_for(graph)

    def fail(*args, **kwargs):
        pytest.fail("A* must not estimate a scale automatically")

    monkeypatch.setattr(path_heuristics, "estimate_context_scale", fail)
    result = graph.compute_path(10, 40, a_star=True, coordinates=coordinates, heuristic_scale=0.1)
    assert result.milepost[-1] == 2


def test_haversine_antipodes_and_antimeridian():
    context = make_context([0, 1, 2, 2], [1, 2], [1, 2])
    heuristic = HaversineContext([179.99, -179.99, 0.01], [0, 0, 0], 1)
    scale = estimate_context_scale(context, heuristic)
    assert np.isfinite(scale) and scale > 0
    heuristic = HaversineContext([179.99, -179.99, 0.01], [0, 0, 0], scale)
    results = run(context, 0, 2, heuristic)
    assert results.path_cost_to(2) == 3
    assert_state_tree(context, results)


@pytest.mark.parametrize("coordinates", [([0, 1], [0]), ([np.nan], [0]), ([[1]], [[2]])])
def test_internal_coordinate_checks(coordinates):
    for cls in [EuclideanContext, HaversineContext]:
        with pytest.raises(ValueError):
            cls(*coordinates, 1)
    with pytest.raises(ValueError, match="degrees"):
        HaversineContext([0], [91], 1)


def test_wrapper_checks_queries_and_dimensions():
    context = history_context()
    results = allocate_results(context)
    heuristic = heuristic_for(context, "euclidean")
    with pytest.raises(ValueError, match="only target"):
        a_star(context, SearchQuery(4, 0), 3, heuristic, results)
    with pytest.raises(ValueError, match="only target"):
        a_star(context, SearchQuery(4, 0, np.array([False, True, True, False])), 3, heuristic, results)
    with pytest.raises(ValueError, match="dimensions"):
        run(context, 0, 3, EuclideanContext([0], [0], 1))
    with pytest.raises(ValueError, match="heap must be"):
        run(context, 0, 3, heuristic, heap="invalid")
