"""Route choice keeps a route's turn labels after each search is reused."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Graph
from aequilibrae.paths.route_choice import RouteChoice


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
