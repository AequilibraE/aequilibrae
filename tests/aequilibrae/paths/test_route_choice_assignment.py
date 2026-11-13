"""Path-size logit, imported routes and link/select-link loading."""

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet

from aequilibrae.matrix import from_dict
from aequilibrae.paths import Network, RouteChoice

from .route_choice_helpers import (
    chain_graph,
    configured_choice,
    diamond,
    mapped_network,
    recompute_imported_routes,
    single_od_demand,
)


@pytest.mark.parametrize("cost,beta", [("distance", 0.001), ("free_flow_time", 0.1)])
@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_sioux_falls_assignment(route_choice_setup, cost, beta, algorithm):
    project = route_choice_setup["project"]
    graph = route_choice_setup["graph"]
    graph.set_costs(project.network.build().field_values(cost))

    choice = RouteChoice(graph)
    choice.set_cores(1)
    choice.set_choice_set_generation(algorithm, max_routes=4, max_depth=5, penalty=2.0, beta=beta)

    # These ODs exercise links 23 (9->5), 26 (10->9), and routes avoiding both.
    flows = {(20, 4): 2.0, (21, 2): 3.0, (10, 8): 4.0, (17, 9): 5.0, (5, 3): 6.0}
    demand = from_dict(
        {"flow": np.zeros((graph.num_zones, graph.num_zones)), "zero": np.zeros((graph.num_zones, graph.num_zones))},
        index=graph.centroids,
    )
    for (origin, destination), flow in flows.items():
        demand.mat["flow", origin, destination] = flow

    choice.set_demand(demand)
    choice.set_select_links({"either": [(23, 1), (26, 1)], "both": [[(23, 1), (26, 1)]]})
    choice.execute()

    rows = choice.get_results()
    assert set(zip(rows["origin id"], rows["destination id"], strict=True)) == set(flows)
    np.testing.assert_allclose(rows.groupby(["origin id", "destination id"])["probability"].sum(), 1.0)
    assert (rows.groupby(["origin id", "destination id"]).size() > 1).all()

    source_links = project.network.links.data.set_index("link_id")
    expected_loads = pd.DataFrame(0.0, index=source_links.index, columns=["all", "either", "both"])
    expected_ods = {name: pd.DataFrame(0.0, index=demand.index, columns=demand.index) for name in ["either", "both"]}

    # Walk reported paths independently, checking every link rather than only a load sum or union.
    for _, row in rows.iterrows():
        route = row["route set"]
        np.testing.assert_array_equal(row["route directions"], 1)
        assert row["cost"] == pytest.approx(source_links.loc[route, cost].sum())

        flow = flows[row["origin id"], row["destination id"]] * row["probability"]
        expected_loads.loc[route, "all"] += flow

        selected = {"either": bool(set(route) & {23, 26}), "both": {23, 26}.issubset(route)}
        for name, matches in selected.items():
            if matches:
                expected_loads.loc[route, name] += flow
                expected_ods[name].loc[row["origin id"], row["destination id"]] += flow

    loads = choice.get_load_results()
    assert set(loads.columns) == {f"{field}_{direction}" for direction in ["ab", "ba", "tot"] for field in demand}
    np.testing.assert_allclose(loads["flow_ab"], expected_loads.loc[loads.index, "all"])
    np.testing.assert_allclose(loads["flow_tot"], loads["flow_ab"])
    np.testing.assert_array_equal(loads[["flow_ba", "zero_ab", "zero_ba", "zero_tot"]], 0.0)

    selected_loads = choice.get_select_link_loading_results()
    selected_ods = choice.get_select_link_od_matrix_results()
    for name in expected_ods:
        assert expected_loads[name].sum() > 0
        np.testing.assert_allclose(selected_loads[f"flow_{name}_tot"], expected_loads.loc[loads.index, name])
        np.testing.assert_array_equal(selected_loads[f"flow_{name}_ba"], 0.0)
        np.testing.assert_array_equal(selected_loads[f"zero_{name}_tot"], 0.0)

        np.testing.assert_array_equal(selected_ods[name].index, demand.index)
        np.testing.assert_allclose(selected_ods[name]["flow"].toarray(), expected_ods[name].to_numpy())
        np.testing.assert_array_equal(selected_ods[name]["zero"].toarray(), 0.0)


@pytest.mark.parametrize("recompute_psl", [False, True])
def test_assign_from_df_omits_ods_without_demand(route_choice_setup, recompute_psl):
    graph = route_choice_setup["graph"]
    choice = RouteChoiceSet(graph)

    demand = from_dict({"flow": np.zeros((graph.num_zones, graph.num_zones))}, index=graph.centroids)
    demand.mat["flow", 1, 2] = 2.0
    demand.mat["flow", 1, 3] = 3.0

    # The final two valid single-link routes have no demand and must not appear in the results.
    supplied = pd.DataFrame(
        {
            "origin id": [1, 1, 2, 9],
            "destination id": [2, 3, 6, 5],
            "route set": [[1], [2], [4], [23]],
            "route directions": [[1]] * 4,
            "probability": [0.25, 0.5, 0.75, 1.0],
        }
    )

    choice.assign_from_df(supplied, demand, select_links={}, recompute_psl=recompute_psl)
    result = choice.get_results()
    assert result[["origin id", "destination id"]].to_numpy().tolist() == [[1, 2], [1, 3]]

    expected_loads = np.zeros(graph.mapping.network.link_count)
    expected_loads[graph.mapping.network.link_index([1, 2], 1)] = [2.0, 3.0] if recompute_psl else [0.5, 1.5]
    np.testing.assert_allclose(choice.get_link_loading()["flow"], expected_loads)
    np.testing.assert_array_equal(result["mask"], True)

    if recompute_psl:
        links = route_choice_setup["project"].network.links.data.set_index("link_id")
        np.testing.assert_allclose(result["cost"], links.loc[[1, 2], "distance"])
        np.testing.assert_array_equal(result[["path overlap", "probability"]], 1.0)
    else:
        np.testing.assert_array_equal(result[["cost", "path overlap"]], 0.0)
        np.testing.assert_array_equal(result["probability"], [0.25, 0.5])


def test_shared_turn_cost_counts_towards_overlap():
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14, 15],
            "a_node": [10, 20, 30, 30, 40],
            "b_node": [20, 30, 50, 40, 50],
            "direction": [1] * 5,
            "time": [1.0] * 5,
        }
    )
    network = Network(
        links,
        centroids=np.array([10, 20, 30, 40, 50]),
        turn_restrictions=pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [30], "penalty": [3.0]}),
    )
    graph = network.build_graph("c", cost="time", compact=True, block_centroid_flows=False)

    choice = configured_choice(graph)
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.execute_single(10, 50, demand=1.0)

    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[11, 12, 13]["cost"] == pytest.approx(6.0)
    assert table[11, 12, 13]["path overlap"] == pytest.approx(3.5 / 6.0)
    assert table[11, 12, 14, 15]["cost"] == pytest.approx(7.0)
    assert table[11, 12, 14, 15]["path overlap"] == pytest.approx(4.5 / 7.0)

    imported = recompute_imported_routes(graph, [[11, 12, 13], [11, 12, 14, 15]], destination=50)
    imported_table = {tuple(row["route set"]): row for _, row in imported.iterrows()}
    for route, generated in table.items():
        for field in ("cost", "path overlap", "probability"):
            assert imported_table[route][field] == pytest.approx(generated[field])


def test_recomputed_psl_uses_turn_penalties():
    rows = recompute_imported_routes(diamond(5.0), [[71, 12], [55, 24]])
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}

    assert table[55, 24]["cost"] == 4.0
    assert table[71, 12]["cost"] == 7.0
    assert table[71, 12]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(3.0)))


@pytest.mark.parametrize("policy", ["node-based", "implicit-ban", "explicit-allow"])
def test_recomputed_psl_applies_uturn_rules(policy):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 20, 10, 20],
            "b_node": [20, 10, 40, 40],
            "direction": [1] * 4,
            "time": [1.0] * 4,
        }
    )

    turns = None
    if policy != "node-based":
        turns = pd.DataFrame(
            {
                "from_node": [10],
                "via_node": [20],
                "to_node": [10 if policy == "explicit-allow" else 40],
                "penalty": [0.0],
            }
        )
    graph = Network(links, centroids=[10, 20, 40], turn_restrictions=turns).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=False
    )

    if policy == "implicit-ban":
        with pytest.raises(ValueError, match="disallowed U-turn"):
            recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
    else:
        # Node-based routing does not impose an additional U-turn policy on imports.
        result = recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
        assert result["mask"].all()
        np.testing.assert_allclose(result["cost"], [3.0, 2.0])


@pytest.mark.parametrize("recompute", [False, True])
def test_imported_mask_excludes_loading_and_select_links(recompute):
    choice = configured_choice(diamond())
    choice.set_select_links({"excluded": [[(71, 1)]], "used": [[(55, 1)]]})
    choice.set_demand(single_od_demand(10, 40, flow=10.0))

    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "route directions": [[1, 1], [1, 1]],
            "cost": [123.0, 456.0],
            "mask": [False, True],
            "probability": [0.9, 0.1],
        }
    )

    choice.execute_from_pandas(supplied, recompute_psl=recompute)
    result = choice.get_results()

    np.testing.assert_allclose(result["cost"], [2.5, 4.0] if recompute else [123.0, 456.0])

    # Supplied probabilities are not renormalised unless PSL is recomputed.
    np.testing.assert_allclose(result["probability"], [0.0, 1.0 if recompute else 0.1])

    expected = 10.0 if recompute else 1.0
    np.testing.assert_allclose(choice.get_load_results()["flow_tot"], [0.0, 0.0, expected, expected])

    selected = choice.get_select_link_loading_results()
    assert (selected["flow_excluded_tot"] == 0.0).all()
    np.testing.assert_allclose(selected["flow_used_tot"], [0.0, 0.0, expected, expected])


@pytest.mark.parametrize("recompute_psl", [False, True])
def test_empty_route_table_loads_no_demand(recompute_psl):
    choice = configured_choice(diamond())
    choice.set_demand(single_od_demand(10, 40))

    supplied = pd.DataFrame(columns=["origin id", "destination id", "route set", "route directions", "probability"])
    # Empty integer columns are required by the route identity interface.
    supplied = supplied.astype({"origin id": "uint64", "destination id": "uint64", "probability": "float64"})

    choice.execute_from_pandas(supplied, recompute_psl=recompute_psl)

    assert choice.get_results().empty
    np.testing.assert_array_equal(choice.get_load_results()["flow_tot"], 0.0)


def test_recomputed_psl_replaces_supplied_costs_and_preserves_exclusions():
    choice = RouteChoice(diamond(0.0))
    choice.set_choice_set_generation(disutility_cutoff_constant=0, disutility_cutoff_coefficient=1)

    supplied = pd.DataFrame(
        {
            "origin id": [10, 10, 10],
            "destination id": [40, 40, 40],
            "route set": [[71, 12], [55, 24], [55, 24]],
            "route directions": [[1, 1], [1, 1], [1, 1]],
            "cost": [np.nan, -1.0, np.inf],
            "mask": [False, True, False],
        },
        index=[5, 5, 0],
    )
    before = supplied.copy(deep=True)

    result = choice.recompute_psl(supplied)

    pd.testing.assert_frame_equal(supplied, before)
    assert result.index.tolist() == [5, 5, 0]
    assert result["cost"].tolist() == [2.0, 4.0, 4.0]
    assert result["mask"].tolist() == [False, True, False]
    assert result["path overlap"].tolist() == [0.0, 1.0, 0.0]
    assert result["probability"].tolist() == [0.0, 1.0, 0.0]


def test_route_choice_keeps_borrowed_graph_alive():
    graph = diamond()
    choice = configured_choice(graph)
    del graph

    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.execute_single(10, 40, demand=1.0)
    assert sorted(choice.get_results()["cost"]) == [2.5, 4.0]

    supplied = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": [[71, 12]], "route directions": [[1, 1]]}
    )
    assert choice.recompute_psl(supplied)["cost"].iloc[0] == 2.5


def test_complete_expansion_traversals_are_not_deduplicated():
    # Use a connected cycle so the repeated traversal is a valid imported path.
    network = Network(
        pd.DataFrame(
            {
                "link_id": [11, 12, 13, 14],
                "a_node": [10, 20, 30, 40],
                "b_node": [20, 30, 40, 10],
                "direction": 1,
                "time": 1.0,
            }
        ),
        centroids=[10, 40],
    )
    choice = configured_choice(network.build_graph("c", "time", compact=True, block_centroid_flows=False))
    choice.set_demand(single_od_demand(10, 40, flow=2.0))

    route = [11, 12, 13, 14, 11, 12, 13]
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [route],
            "route directions": [[1] * len(route)],
            "probability": [1.0],
        }
    )

    choice.execute_from_pandas(supplied)

    assert choice.get_load_results()["flow_tot"].tolist() == [4.0, 4.0, 4.0, 2.0]
    assert choice.get_results()["route set"].iloc[0].tolist() == route


def test_costs_are_snapshotted_for_search_and_imports():
    graph = chain_graph()
    choice = configured_choice(graph)

    graph.set_costs(np.array([10.0, 20.0, 30.0]))
    choice.execute_single(10, 40, demand=1.0)
    result = choice.get_results()

    assert result["cost"].iloc[0] == 6.0
    assert choice.recompute_psl(result)["cost"].iloc[0] == 6.0

    later = configured_choice(graph)
    later.execute_single(10, 40, demand=1.0)
    assert later.get_results()["cost"].iloc[0] == 60.0


def test_imported_routes_use_prepared_cost_field():
    choice = RouteChoice(diamond(5.0, cost="distance"))

    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "route directions": [[1, 1], [1, 1]],
        }
    )

    result = choice.recompute_psl(supplied)

    assert result["cost"].tolist() == [75.0, 110.0]
    assert result["path overlap"].tolist() == [1.0, 1.0]
    assert result["probability"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("turns", [False, True])
def test_masked_first_route_does_not_hide_overlap(turns):
    restrictions = (
        pd.DataFrame({"from_node": [150], "via_node": [200], "to_node": [300], "penalty": [0.5]}) if turns else None
    )
    graph = mapped_network().build_graph(
        "c", "time", compact=True, remove_dead_ends=True, turn_restrictions=restrictions
    )

    supplied = pd.DataFrame(
        {
            "origin id": [100, 100],
            "destination id": [300, 300],
            "route set": [[900, 0, 81], [900, 0, 81]],
            "route directions": [[1, -1, 1], [1, -1, 1]],
            "mask": [False, True],
        }
    )

    result = configured_choice(graph).recompute_psl(supplied)

    assert result["mask"].tolist() == [False, True]
    assert result["cost"].iloc[1] == (8.5 if turns else 8.0)
    assert result["path overlap"].tolist() == [0.0, 1.0]
    assert result["probability"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "compact,remove_dead_ends,turns,cores,entry_point",
    [
        (False, False, False, 1, "generate"),
        (True, False, True, 2, "generate"),
        (True, True, False, 1, "import"),
        (True, True, True, 1, "recompute"),
    ],
)
def test_loading_preserves_network_identity(compact, remove_dead_ends, turns, cores, entry_point):
    restrictions = (
        pd.DataFrame({"from_node": [150], "via_node": [200], "to_node": [300], "penalty": [0.5]}) if turns else None
    )
    graph = mapped_network().build_graph(
        "c", "time", compact=compact, remove_dead_ends=remove_dead_ends, turn_restrictions=restrictions
    )
    costs = graph.cost.copy()

    demand = from_dict(
        {"flow": np.array([[0.0, 2.0], [4.0, 0.0]]), "extra": np.array([[0.0, 20.0], [40.0, 0.0]])},
        index=np.array([100, 300]),  # Demand order differs from the graph's centroid prefix.
    )

    choice = configured_choice(graph)
    choice.set_cores(cores)
    choice.set_demand(demand)
    with pytest.warns(UserWarning, match="OR alternatives"):
        choice.set_select_links({"both": [(0, 0)], "forward": [(0, -1)]})

    if entry_point == "generate":
        choice.execute()
    else:
        supplied = pd.DataFrame(
            {
                "origin id": [100, 300],
                "destination id": [300, 100],
                "route set": [[900, 0, 81], [81, 0, 900]],
                "route directions": [[1, -1, 1], [-1, 1, -1]],
                "probability": [1.0, 1.0],
                "cost": [8.0, 10.0],
            }
        )
        choice.execute_from_pandas(supplied, recompute_psl=entry_point == "recompute")

    rows = choice.get_results().sort_values(["origin id", "destination id"])
    assert rows[["origin id", "destination id"]].to_numpy().tolist() == [[100, 300], [300, 100]]
    assert [list(route) for route in rows["route set"]] == [[900, 0, 81], [81, 0, 900]]
    assert [list(route) for route in rows["route directions"]] == [[1, -1, 1], [-1, 1, -1]]
    np.testing.assert_allclose(rows["cost"], [8.5 if turns else 8.0, 10.0])
    np.testing.assert_array_equal(graph.cost, costs)

    loads = choice.get_load_results()
    assert loads.index.tolist() == [0, 900, 81, 66, 75]
    assert loads["flow_ab"].tolist() == [4.0, 2.0, 2.0, 0.0, 0.0]
    assert loads["flow_ba"].tolist() == [2.0, 4.0, 4.0, 0.0, 0.0]
    np.testing.assert_allclose(loads["extra_tot"], loads["flow_tot"] * 10)

    selected = choice.get_select_link_loading_results()
    np.testing.assert_allclose(selected["flow_both_tot"], loads["flow_tot"])
    assert selected["flow_forward_ab"].tolist() == [0.0, 2.0, 2.0, 0.0, 0.0]
    assert selected["flow_forward_ba"].tolist() == [2.0, 0.0, 0.0, 0.0, 0.0]

    matrices = choice.get_select_link_od_matrix_results()
    np.testing.assert_array_equal(matrices["forward"]["flow"].toarray(), [[0.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(matrices["forward"].index, demand.index)
