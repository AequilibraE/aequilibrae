"""Route choice respects turn controls and keeps turn labels after each search is reused."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Graph
from aequilibrae.matrix import GeneralisedCOODemand
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.paths.route_choice import RouteChoice
from aequilibrae.utils.cython.bridge import Bridge


def junction(penalty):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 10, 30, 20],
            "b_node": [20, 30, 20, 40],
            "direction": [1] * 4,
            "time": [1.0, 2.0, 1.0, 1.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return graph


def batched_junction_routes(graph, bfsle, max_routes):
    ods = [(10, 40), (20, 40)]
    demand = GeneralisedCOODemand(
        "origin id", "destination id", graph.nodes_to_indices, shape=(graph.num_zones, graph.num_zones)
    )
    demand.add_df(
        pd.DataFrame(
            {"flow": [1.0, 1.0]},
            index=pd.MultiIndex.from_tuples(ods, names=["origin id", "destination id"]),
        )
    )
    choice = RouteChoiceSet(graph)
    with Bridge() as bridge:
        choice.batched(
            demand,
            max_routes=max_routes,
            max_depth=6,
            bfsle=bfsle,
            penalty=4.0,
            path_size_logit=True,
            cores=1,
            bridge=bridge,
        )
    return {
        (row["origin id"], row["destination id"], tuple(row["route set"])): row
        for _, row in choice.get_results().iterrows()
    }


@pytest.mark.parametrize("bfsle", [True, False])
def test_batched_prohibited_turn_keeps_alternative_arrival(bfsle):
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


def diamond(penalty=0.5):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [30.0, 40.0, 50.0, 60.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return graph


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_generated_routes_keep_original_cost_and_turns(algorithm):
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)

    # Search may penalise links, but PSL must use their original costs.
    routes = choice.execute_single(10, 40, demand=1.0)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert set(routes) == {(71, 12), (55, 24)}
    assert table[71, 12]["cost"] == pytest.approx(2.5)
    assert table[55, 24]["cost"] == pytest.approx(4.0)
    assert table[71, 12]["path overlap"] == pytest.approx(1.0)
    assert table[55, 24]["path overlap"] == pytest.approx(1.0)
    assert table[71, 12]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(-1.5)))


def test_shared_turn_cost_counts_towards_overlap():
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14, 15],
            "a_node": [10, 20, 30, 30, 40],
            "b_node": [20, 30, 50, 40, 50],
            "direction": [1] * 5,
            "time": [1.0] * 5,
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40, 50]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_turn_restrictions(
        pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [30], "penalty": [3.0]})
    )
    choice = RouteChoice(graph)
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.execute_single(10, 50, demand=1.0)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[11, 12, 13]["cost"] == pytest.approx(6.0)
    assert table[11, 12, 13]["path overlap"] == pytest.approx(3.5 / 6.0)
    assert table[11, 12, 14, 15]["cost"] == pytest.approx(7.0)
    assert table[11, 12, 14, 15]["path overlap"] == pytest.approx(4.5 / 7.0)


def test_prohibited_turn_is_not_generated():
    choice = RouteChoice(diamond(np.inf))
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    assert choice.execute_single(10, 40, demand=1.0) == [(55, 24)]
    assert choice.get_results()["cost"].iloc[0] == pytest.approx(4.0)


def test_generation_without_psl_does_not_need_turn_steps():
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    assert set(choice.execute_single(10, 40)) == {(71, 12), (55, 24)}
    assert "cost" not in choice.get_results()


def test_route_choice_keeps_context_snapshot():
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.execute_single(10, 40, demand=1.0)
    graph.set_graph("distance")
    graph.compact_nodes_to_indices[:] = -1
    graph.graph.loc[:, "link_id"] = 999
    graph.graph.loc[:, "__compressed_id__"] = graph.compact_num_links
    choice.execute_single(10, 40, demand=1.0)
    assert sorted(choice.get_results()["cost"]) == [2.5, 4.0]
    assert {tuple(route) for route in choice.get_results()["route set"]} == {(71, 12), (55, 24)}
