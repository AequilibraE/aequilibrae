"""Route generation, turn controls and A* dispatch."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import from_df
from aequilibrae.paths import Network, RouteChoice, estimate_heuristic_scale

from .route_choice_helpers import (
    batched_junction_routes,
    compressed_graph,
    coordinates_for,
    diamond,
    junction,
    single_od_demand,
    sorted_results,
)


@pytest.mark.parametrize("nodes,expected", [([1, 2], [(1, 2), (2, 1)]), ([(1, 2)], [(1, 2)])])
def test_prepare(route_choice_setup, nodes, expected):
    choice = route_choice_setup["rc"]
    choice.set_choice_set_generation("bfsle", max_routes=1)
    choice.prepare(nodes)
    choice.execute(perform_assignment=False)
    assert list(choice.get_results()[["origin id", "destination id"]].itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation", "link-penalization", "lp"])
def test_algorithm_aliases_and_parameter_reset(route_choice_setup, algorithm):
    choice = route_choice_setup["rc"]
    choice.set_choice_set_generation(algorithm, max_routes=20, penalty=3.0)
    assert choice.parameters["penalty"] == 3.0
    choice.set_choice_set_generation(algorithm, max_routes=2)
    assert choice.algorithm == ("bfsle" if algorithm == "bfsle" else "link-penalisation")
    assert choice.parameters["max_routes"] == 2
    assert choice.parameters["penalty"] == (1.0 if algorithm == "bfsle" else 1.01)


@pytest.mark.parametrize(
    "compact,turn,algorithm,a_star",
    [
        (False, False, "bfsle", False),
        (True, False, "link-penalisation", False),
        (True, True, "bfsle", True),
        (True, True, "link-penalisation", True),
    ],
)
def test_search_uses_routing_node_order(compact, turn, algorithm, a_star):
    links = pd.DataFrame(
        {
            "link_id": np.arange(1, 9),
            "a_node": [1, 2, 3, 3, 4, 4, 5, 6],
            "b_node": [2, 3, 4, 5, 5, 6, 6, 7],
            "direction": 0,
            "time": 1.0,
        }
    )
    turns = pd.DataFrame({"from_node": [2], "via_node": [3], "to_node": [4], "penalty": [0.5]}) if turn else None
    graph = Network(links, centroids=[1, 7], turn_restrictions=turns).build_graph(
        "c", cost="time", compact=compact, block_centroid_flows=False
    )
    if compact:
        # Contracting node 2 shifts retained non-centroid indices, unlike centroid indices.
        np.testing.assert_array_equal(graph.mapping.network.node_ids, [1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(graph.all_nodes, [1, 7, 3, 4, 5, 6])
    coordinates = pd.DataFrame({"x": graph.all_nodes.astype(float), "y": 0.0}, index=graph.all_nodes).iloc[::-1]
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_cores(1)
    choice.set_choice_set_generation(
        algorithm, max_routes=1, max_depth=1, a_star=a_star, heuristic_scale=0.5 if a_star else None
    )
    # Node 3 has an in-range but wrong full index; node 6's full index is out of routing bounds.
    expected = {
        (1, 3): ((1, 2), (1, 1)),
        (3, 1): ((2, 1), (-1, -1)),
        (6, 7): ((8,), (1,)),
        (7, 6): ((8,), (-1,)),
    }
    for (origin, destination), (ids, directions) in expected.items():
        routes = choice.execute_single(origin, destination)
        assert len(routes) == 1
        np.testing.assert_array_equal(routes[0][0], ids)
        np.testing.assert_array_equal(routes[0][1], directions)

    # Batch routing shares the lookup across workers and must map both endpoints correctly.
    choice.set_cores(2)
    choice.prepare(list(expected))
    choice.execute(perform_assignment=False)
    rows = choice.get_results()
    actual = {
        (row["origin id"], row["destination id"]): (tuple(row["route set"]), tuple(row["route directions"]))
        for _, row in rows.iterrows()
    }
    assert actual == expected

    # Imported routes use network-node indices when validating expanded links.
    recosted = choice.recompute_psl(rows)
    assert recosted["mask"].all()
    np.testing.assert_allclose(recosted["cost"], rows["route set"].map(len))


@pytest.mark.parametrize("bfsle", [True, False])
def test_batched_with_prohibited_turn(bfsle):
    # The direct arrival at 20 is cheaper, but it cannot continue to 40.
    rows = batched_junction_routes(junction(np.inf), bfsle, max_routes=2)
    assert set(rows) == {(10, 40, (12, 13, 14)), (20, 40, (14,))}
    assert rows[10, 40, (12, 13, 14)]["cost"] == pytest.approx(4.0)
    assert rows[20, 40, (14,)]["cost"] == pytest.approx(1.0)


@pytest.mark.parametrize("bfsle", [True, False])
def test_batched_finite_turn_penalty_changes_route_and_cost(bfsle):
    graph = junction(5.0)
    # The direct arrival at 20 costs 1, but its continuation costs 1 + 5.
    shortest = batched_junction_routes(graph, bfsle, max_routes=1)
    assert set(shortest) == {(10, 40, (12, 13, 14)), (20, 40, (14,))}
    assert shortest[10, 40, (12, 13, 14)]["cost"] == pytest.approx(4.0)
    assert shortest[20, 40, (14,)]["cost"] == pytest.approx(1.0)

    # The direct route may be found later. Its reported cost must use the
    # original link costs and include the turn penalty exactly once.
    rows = batched_junction_routes(graph, bfsle, max_routes=2)
    assert set(rows) == {(10, 40, (12, 13, 14)), (10, 40, (11, 14)), (20, 40, (14,))}
    assert rows[10, 40, (12, 13, 14)]["cost"] == pytest.approx(4.0)
    assert rows[10, 40, (11, 14)]["cost"] == pytest.approx(7.0)
    assert rows[20, 40, (14,)]["cost"] == pytest.approx(1.0)


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_psl_uses_unpenalised_link_costs_and_turn_costs(algorithm):
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)

    # Search multiplies link costs by penalty=3.0 to find alternatives. PSL must use
    # unpenalised link costs and include the 0.5 turn cost exactly once.
    routes = choice.execute_single(10, 40, demand=1.0)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert {(tuple(ids), tuple(directions)) for ids, directions in routes} == {
        ((71, 12), (1, 1)),
        ((55, 24), (1, 1)),
    }
    assert table[71, 12]["cost"] == pytest.approx(2.5)
    assert table[55, 24]["cost"] == pytest.approx(4.0)
    assert table[71, 12]["path overlap"] == pytest.approx(1.0)
    assert table[55, 24]["path overlap"] == pytest.approx(1.0)
    assert table[71, 12]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(-1.5)))


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_generation_without_psl_returns_routes_only(algorithm):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)
    routes = choice.execute_single(10, 40)
    assert {(tuple(ids), tuple(directions)) for ids, directions in routes} == {
        ((71, 12), (1, 1)),
        ((55, 24), (1, 1)),
    }
    assert "cost" not in choice.get_results()


@pytest.mark.parametrize(
    "algorithm,penalty,heuristic,turn_penalty,cores",
    [
        ("bfsle", 1.0, "euclidean", None, 1),
        ("bfsle", 4.0, "haversine", 5.0, 2),
        ("bfsle", 1.0, "euclidean", np.inf, 2),
        ("link-penalisation", 4.0, "haversine", None, 2),
        ("link-penalisation", 4.0, "euclidean", 5.0, 1),
        ("link-penalisation", 4.0, "haversine", np.inf, 1),
    ],
)
def test_batched_a_star_matches_dijkstra(algorithm, penalty, heuristic, turn_penalty, cores):
    lonlat = pd.DataFrame({"lon": np.arange(4) * 0.01, "lat": np.zeros(4)}, index=[10, 20, 30, 40])
    graph = junction(turn_penalty, lonlat=lonlat)
    coordinates = coordinates_for(graph, heuristic)
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
        choice.set_demand(
            from_df(
                pd.DataFrame(
                    {
                        "origin": [10, 20, 40, 10],
                        "destination": [40, 40, 10, 10],
                        "flow": [2.0, 3.0, 4.0, 5.0],
                    }
                ),
                "origin",
                "destination",
                sparse=True,
            )
        )
        choice.set_select_links({"selected": [(14, 1)]})
        choice.execute()
        choices.append(choice)
    reference, actual = choices
    pd.testing.assert_frame_equal(sorted_results(actual), sorted_results(reference), check_like=True)
    pd.testing.assert_frame_equal(actual.get_load_results(), reference.get_load_results(), check_like=True)
    pd.testing.assert_frame_equal(
        actual.get_select_link_loading_results(), reference.get_select_link_loading_results(), check_like=True
    )


@pytest.mark.parametrize(
    "algorithm,heuristic,turn", [("bfsle", "euclidean", False), ("link-penalisation", "haversine", True)]
)
def test_compact_coordinates_and_expanded_routes(algorithm, heuristic, turn):
    graph = compressed_graph(turn)
    coordinates = coordinates_for(graph, heuristic)
    scale = estimate_heuristic_scale(graph, coordinates, heuristic=heuristic)
    coordinates = coordinates.loc[graph.all_nodes].iloc[::-1]
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_cores(1)
    options = {"max_routes": 2, "max_depth": 6, "penalty": 4.0}
    choice.set_choice_set_generation(algorithm, **options)
    choice.execute_single(10, 60, demand=1.0)
    reference = sorted_results(choice)

    # Coordinate changes after construction must not change this snapshot.
    coordinates.loc[:, :] = np.nan
    for scale_value in [scale, 0.0]:
        choice.set_choice_set_generation(
            algorithm, **options, a_star=True, heuristic=heuristic, heuristic_scale=scale_value
        )
        routes = choice.execute_single(10, 60, demand=1.0)
        assert {(tuple(ids), tuple(directions)) for ids, directions in routes} == {
            ((71, 72, 12), (1, 1, 1)),
            ((55, 56, 24), (1, 1, 1)),
        }
        pd.testing.assert_frame_equal(sorted_results(choice), reference, check_like=True)

    # A later Dijkstra execution must not retain the A* selection.
    choice.set_choice_set_generation(algorithm, **options)
    choice.execute_single(10, 60, demand=1.0)
    pd.testing.assert_frame_equal(sorted_results(choice), reference, check_like=True)


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("heuristic,turn", [("euclidean", False), ("haversine", True)])
def test_explicit_scale_controls_a_star_search(algorithm, heuristic, turn):
    coordinates = pd.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "y": [0.0] * 4}, index=[10, 20, 30, 40])
    graph = diamond(0.5 if turn else None, lonlat=coordinates.rename(columns={"x": "lon", "y": "lat"}))
    choice = RouteChoice(
        graph,
        coordinates=coordinates if heuristic == "euclidean" else coordinates.rename(columns={"x": "lon", "y": "lat"}),
    )
    choice.set_cores(1)
    choice.set_choice_set_generation(algorithm, max_routes=1)
    routes = choice.execute_single(10, 40)
    assert len(routes) == 1
    np.testing.assert_array_equal(routes[0][0], [71, 12])
    np.testing.assert_array_equal(routes[0][1], [1, 1])
    # This deliberately unsafe scale favours the more expensive branch. It also
    # checks that A* is actually dispatched, rather than silently using Dijkstra.
    choice.set_choice_set_generation(algorithm, max_routes=1, a_star=True, heuristic=heuristic, heuristic_scale=100.0)
    routes = choice.execute_single(10, 40)
    assert len(routes) == 1
    np.testing.assert_array_equal(routes[0][0], [55, 24])
    np.testing.assert_array_equal(routes[0][1], [1, 1])


@pytest.mark.parametrize("algorithm,a_star", [("bfsle", False), ("link-penalisation", True)])
def test_empty_graph_returns_no_routes_or_flows(algorithm, a_star):
    network = Network(
        pd.DataFrame({"link_id": [9], "a_node": [20], "b_node": [30], "direction": [1], "time": [1.0]}),
        centroids=[10, 40],
    )
    graph = network.build_graph("c", "time", remove_dead_ends=True)
    assert graph.num_links == 0
    coordinates = pd.DataFrame({"x": [0.0, 1.0], "y": 0.0}, index=[10, 40])
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_cores(1)
    choice.set_choice_set_generation(algorithm, max_routes=1, a_star=a_star, heuristic_scale=0.0)
    assert choice.execute_single(10, 40, demand=1.0) == []
    choice.set_demand(single_od_demand(10, 40, flow=2.0))
    choice.execute()
    assert choice.get_results().empty
    assert choice.get_load_results()["flow_tot"].tolist() == [0.0]
    choice.execute_from_pandas(choice.get_results(), recompute_psl=True)
    assert choice.get_results().empty
    assert choice.get_load_results()["flow_tot"].tolist() == [0.0]
