"""A* route generation on compact graphs with optional turn controls."""

import json

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph, RouteChoice, estimate_heuristic_scale
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.cython.bridge import Bridge

from .test_route_choice_routing_context import diamond, junction


def coordinates_for(graph):
    coordinates = pd.DataFrame(
        {"x": np.arange(graph.num_nodes, dtype=float), "y": np.zeros(graph.num_nodes)},
        index=graph.all_nodes,
    )
    graph.lonlat_index = coordinates.rename(columns={"x": "lon", "y": "lat"}) * 0.01
    # External IDs, not row order, determine the compact coordinate mapping.
    return coordinates.iloc[::-1]


def sorted_results(choice):
    result = choice.get_results().copy()
    result["route set"] = result["route set"].map(tuple)
    return result.sort_values(["origin id", "destination id", "route set"]).reset_index(drop=True)


@pytest.mark.parametrize("algorithm,penalty", [("bfsle", 1.0), ("bfsle", 4.0), ("link-penalisation", 4.0)])
@pytest.mark.parametrize("heuristic", ["euclidean", "haversine"])
@pytest.mark.parametrize("turn_penalty", [None, 5.0, np.inf])
@pytest.mark.parametrize("cores", [1, 2])
def test_batched_matches_dijkstra(algorithm, penalty, heuristic, turn_penalty, cores):
    graph = junction(5.0 if turn_penalty is None else turn_penalty)
    if turn_penalty is None:
        graph.clear_turn_restrictions()
    coordinates = coordinates_for(graph)
    scale = estimate_heuristic_scale(graph, coordinates, heuristic=heuristic)
    assert scale > 0
    choices = []
    for a_star in [False, True]:
        choice = RouteChoice(graph, coordinates=coordinates)
        choice.set_cores(cores)
        choice.set_choice_set_generation(
            algorithm,
            max_routes=2,
            max_depth=6,
            penalty=penalty,
            a_star=a_star,
            heuristic=heuristic,
            heuristic_scale=scale,
        )
        choice.add_demand(
            pd.DataFrame(
                {"flow": [2.0, 3.0, 4.0, 5.0]},
                index=pd.MultiIndex.from_tuples(
                    [(10, 40), (20, 40), (40, 10), (10, 10)], names=choice.demand_index_names
                ),
            )
        )
        choice.set_select_links({"selected": [(14, 1)]})
        choice.execute()
        choices.append(choice)
    reference, actual = choices
    pd.testing.assert_frame_equal(sorted_results(actual), sorted_results(reference))
    pd.testing.assert_frame_equal(actual.get_load_results(), reference.get_load_results())
    pd.testing.assert_frame_equal(actual.get_select_link_loading_results(), reference.get_select_link_loading_results())


def compressed_graph(turn):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [71, 72, 12, 55, 56, 24],
            "a_node": [10, 20, 30, 10, 40, 50],
            "b_node": [20, 30, 60, 40, 50, 60],
            "direction": [1] * 6,
            "time": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    graph.prepare_graph(np.array([10, 60]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    if turn:
        graph.set_turn_restrictions(
            pd.DataFrame(
                {
                    "from_node": [20],
                    "via_node": [30],
                    "to_node": [60],
                    "penalty": [5.0],
                }
            )
        )
    assert graph.compact_num_nodes < graph.num_nodes
    return graph


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("heuristic", ["euclidean", "haversine"])
@pytest.mark.parametrize("turn", [False, True])
def test_compact_coordinates_and_expanded_routes(algorithm, heuristic, turn):
    graph = compressed_graph(turn)
    coordinates = coordinates_for(graph)
    scale = estimate_heuristic_scale(graph, coordinates, heuristic=heuristic)
    coordinates = coordinates.loc[graph.compact_all_nodes].iloc[::-1]
    graph.lonlat_index = graph.lonlat_index.loc[graph.compact_all_nodes].iloc[::-1]
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_cores(1)
    options = {"max_routes": 2, "max_depth": 6, "penalty": 4.0}
    choice.set_choice_set_generation(algorithm, **options)
    choice.execute_single(10, 60, demand=1.0)
    reference = sorted_results(choice)

    # Coordinate changes after construction must not change this snapshot.
    coordinates.loc[:, :] = np.nan
    graph.lonlat_index.loc[:, :] = np.nan
    for scale_value in [scale, 0.0]:
        choice.set_choice_set_generation(
            algorithm, **options, a_star=True, heuristic=heuristic, heuristic_scale=scale_value
        )
        routes = choice.execute_single(10, 60, demand=1.0)
        assert set(routes) == {(71, 72, 12), (55, 56, 24)}
        pd.testing.assert_frame_equal(sorted_results(choice), reference)
        json.dumps(choice.parameters)

    # A later Dijkstra execution must not retain the A* selection.
    choice.set_choice_set_generation(algorithm, **options)
    choice.execute_single(10, 60, demand=1.0)
    pd.testing.assert_frame_equal(sorted_results(choice), reference)


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("heuristic", ["euclidean", "haversine"])
@pytest.mark.parametrize("turn", [False, True])
def test_explicit_scale_controls_search(algorithm, heuristic, turn):
    graph = diamond()
    if not turn:
        graph.clear_turn_restrictions()
    coordinates = pd.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "y": [0.0] * 4}, index=[10, 20, 30, 40])
    graph.lonlat_index = coordinates.rename(columns={"x": "lon", "y": "lat"})
    choice = RouteChoice(graph, coordinates=coordinates if heuristic == "euclidean" else None)
    choice.set_cores(1)
    choice.set_choice_set_generation(algorithm, max_routes=1)
    assert choice.execute_single(10, 40) == [(71, 12)]
    # This deliberately unsafe scale favours the more expensive branch. It also
    # checks that A* is actually dispatched, rather than silently using Dijkstra.
    choice.set_choice_set_generation(algorithm, max_routes=1, a_star=True, heuristic=heuristic, heuristic_scale=100.0)
    assert choice.execute_single(10, 40) == [(55, 24)]


@pytest.mark.parametrize("a_star", [False, True])
@pytest.mark.parametrize("penalty", [-1.0, 0.0, 0.99, np.nan, np.inf, -np.inf])
def test_invalid_penalty_rejected_at_both_entry_points(a_star, penalty):
    graph = diamond()
    options = {"a_star": a_star, "heuristic_scale": 0.0, "penalty": penalty, "max_routes": 1}
    with pytest.raises(ValueError, match="penalty.*finite and >= 1"):
        RouteChoice(graph).set_choice_set_generation("bfsle", **options)
    with Bridge() as bridge, pytest.raises(ValueError, match="penalty.*finite and >= 1"):
        RouteChoiceSet(graph).run(10, 40, (4, 4), bridge=bridge, **options)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"a_star": True}, "explicit heuristic_scale"),
        ({"a_star": True, "heuristic_scale": -1.0}, "finite and nonnegative"),
        ({"a_star": True, "heuristic_scale": np.inf}, "finite and nonnegative"),
        ({"a_star": True, "heuristic_scale": np.nan}, "finite and nonnegative"),
        ({"heuristic_scale": -1.0}, "finite and nonnegative"),
        ({"heuristic": "unknown"}, "heuristic must be one of"),
    ],
)
def test_invalid_search_options_rejected_at_both_entry_points(options, match):
    graph = diamond()
    with pytest.raises(ValueError, match=match):
        RouteChoice(graph).set_choice_set_generation("bfsle", max_routes=1, **options)
    with Bridge() as bridge, pytest.raises(ValueError, match=match):
        RouteChoiceSet(graph).run(10, 40, (4, 4), max_routes=1, bridge=bridge, **options)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "requires coordinate columns"),
        ("node", "every graph node ID"),
        ("duplicate", "node IDs must be unique"),
        ("nonfinite", "coordinates must be finite"),
    ],
)
def test_invalid_coordinates(problem, match):
    graph = diamond()
    coordinates = coordinates_for(graph)
    if problem == "missing":
        coordinates = None
    elif problem == "node":
        coordinates = coordinates.drop(20)
    elif problem == "duplicate":
        coordinates = pd.concat([coordinates, coordinates.iloc[:1]])
    else:
        coordinates.loc[20, "x"] = np.nan
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_choice_set_generation("bfsle", max_routes=1, a_star=True, heuristic_scale=0.0)
    with pytest.raises(ValueError, match=match):
        choice.execute_single(10, 40)
