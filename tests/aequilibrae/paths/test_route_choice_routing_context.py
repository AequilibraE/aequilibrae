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
    graph.set_turn_restrictions(pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [30], "penalty": [3.0]}))
    choice = RouteChoice(graph)
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


def test_prohibited_turn_is_not_generated():
    choice = RouteChoice(diamond(np.inf))
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    assert choice.execute_single(10, 40, demand=1.0) == [(55, 24)]
    assert choice.get_results()["cost"].iloc[0] == pytest.approx(4.0)


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_generation_without_psl_does_not_need_turn_steps(algorithm):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)
    assert set(choice.execute_single(10, 40)) == {(71, 12), (55, 24)}
    assert "cost" not in choice.get_results()


def recompute_imported_routes(graph, routes, *, destination=40, log_warnings=True, return_choice=False):
    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, destination)], names=["origin id", "destination id"]),
        )
    )
    supplied = pd.DataFrame(
        {"origin id": [10] * len(routes), "destination id": [destination] * len(routes), "route set": routes}
    )
    choice.execute_from_pandas(supplied, recompute_psl=True, log_warnings=log_warnings)
    return choice if return_choice else choice.get_results()


@pytest.mark.parametrize("penalty", [5.0, np.inf])
def test_recomputed_psl_uses_turn_penalties_and_masks_bans(penalty, caplog):
    choice = recompute_imported_routes(diamond(penalty), [[71, 12], [55, 24]], return_choice=True)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[55, 24]["cost"] == pytest.approx(4.0)
    if np.isfinite(penalty):
        assert table[71, 12]["cost"] == pytest.approx(2.0 + penalty)
        assert table[71, 12]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(penalty - 2.0)))
    else:
        assert np.isinf(table[71, 12]["cost"])
        assert not table[71, 12]["mask"]
        assert table[71, 12]["probability"] == 0.0
        assert table[55, 24]["probability"] == 1.0
        loads = choice.get_load_results()["flow_tot"]
        assert loads.loc[71] == loads.loc[12] == 0.0
        assert loads.loc[55] == loads.loc[24] == 1.0
        assert "prohibited turn" in caplog.text


def test_recomputed_psl_checks_cannot_be_disabled(caplog):
    rows = recompute_imported_routes(diamond(5.0), [[71, 24], [55, 24]], log_warnings=False)
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert np.isinf(table[71, 24]["cost"])
    assert not table[71, 24]["mask"]
    assert table[71, 24]["probability"] == 0.0
    assert table[55, 24]["probability"] == 1.0
    assert "Invalid route" not in caplog.text

    rows = recompute_imported_routes(diamond(np.inf), [[71, 12], [55, 24]], log_warnings=False)
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert not table[71, 12]["mask"]
    assert table[71, 12]["probability"] == 0.0


def test_recomputed_psl_all_routes_banned(caplog):
    graph = diamond(np.inf)
    choice = recompute_imported_routes(graph, [[71, 12]], return_choice=True)
    rows = choice.get_results()
    assert np.isinf(rows["cost"].iloc[0])
    assert not rows["mask"].iloc[0]
    assert rows["probability"].iloc[0] == 0.0
    assert np.all(choice.get_load_results()["flow_tot"] == 0.0)
    assert "prohibited turn" in caplog.text

    rows = recompute_imported_routes(graph, [[71, 12]], log_warnings=False)
    assert np.isinf(rows["cost"].iloc[0])
    assert rows["probability"].iloc[0] == 0.0


@pytest.mark.parametrize("allow_explicit_uturn", [False, True])
def test_recomputed_psl_applies_uturn_rules(allow_explicit_uturn, caplog):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 20, 10, 20],
            "b_node": [20, 10, 40, 40],
            "direction": [1] * 4,
            "time": [1.0] * 4,
        }
    )
    graph.prepare_graph(np.array([10, 20, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    turns = pd.DataFrame(
        {
            "from_node": [10],
            "via_node": [20],
            "to_node": [10 if allow_explicit_uturn else 40],
            "penalty": [0.0],
        }
    )
    graph.set_turn_restrictions(turns)

    rows = recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    if allow_explicit_uturn:
        assert table[11, 12, 13]["cost"] == pytest.approx(3.0)
        assert table[11, 12, 13]["mask"]
    else:
        assert np.isinf(table[11, 12, 13]["cost"])
        assert not table[11, 12, 13]["mask"]
        assert table[11, 12, 13]["probability"] == 0.0
        assert "disallowed U-turn" in caplog.text

        graph.clear_turn_restrictions()
        node_rows = recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
        node_table = {tuple(row["route set"]): row for _, row in node_rows.iterrows()}
        assert not node_table[11, 12, 13]["mask"]
        assert node_table[11, 14]["probability"] == 1.0


def test_recomputed_psl_from_path_files_includes_turns(tmp_path):
    graph = diamond(5.0)
    source = RouteChoice(graph)
    source.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    source.execute_single(10, 40, demand=1.0)
    file = tmp_path / "routes.parquet"
    source.get_results().to_parquet(file, index=False)

    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    choice.execute_from_path_files(file, recompute_psl=True)
    rows = choice.get_results()
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert table[71, 12]["cost"] == pytest.approx(7.0)
    assert table[55, 24]["cost"] == pytest.approx(4.0)


def test_recomputed_psl_validates_node_routes_without_turn_tables(caplog):
    graph = diamond()
    graph.clear_turn_restrictions()
    rows = recompute_imported_routes(graph, [[71, 24], [55, 24]])
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert not table[71, 24]["mask"]
    assert table[55, 24]["probability"] == 1.0
    assert "disconnected links" in caplog.text


def test_recomputed_psl_masks_disconnected_links(caplog):
    rows = recompute_imported_routes(diamond(), [[71, 24], [55, 24]])
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert not table[71, 24]["mask"]
    assert table[71, 24]["probability"] == 0.0
    assert "disconnected links" in caplog.text


def test_imported_link_absent_from_compact_graph_is_rejected():
    graph = diamond()
    # Prepare expansion first so this synthetic missing-link crosswalk does not affect its construction.
    graph.create_compressed_link_network_mapping()
    graph.graph.loc[graph.graph.link_id == 71, "__compressed_id__"] = graph.compact_num_links
    choice = RouteChoiceSet(graph)
    demand = GeneralisedCOODemand(
        "origin id", "destination id", graph.nodes_to_indices, shape=(graph.num_zones, graph.num_zones)
    )
    demand.add_df(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    supplied = pd.DataFrame({"origin id": [10], "destination id": [40], "route set": [[71, 12]]})
    with pytest.raises(ValueError, match="absent from the compact graph"):
        choice.assign_from_df(supplied, demand, select_links={}, recompute_psl=True)


@pytest.mark.parametrize(
    "origin,destination,invalid,valid,reason",
    [
        (20, 40, [55, 24], [12], "starts at the wrong node"),
        (10, 30, [71, 12], [55], "ends at the wrong node"),
    ],
)
def test_imported_route_validation_checks_od_endpoints(origin, destination, invalid, valid, reason, caplog):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(origin, destination)], names=["origin id", "destination id"]),
        )
    )
    df = pd.DataFrame(
        {
            "origin id": [origin, origin],
            "destination id": [destination, destination],
            "route set": [invalid, valid],
            "probability": [0.25, 0.75],
        }
    )
    choice.execute_from_pandas(df, recompute_psl=True)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert np.isinf(table[tuple(invalid)]["cost"])
    assert not table[tuple(invalid)]["mask"]
    assert table[tuple(invalid)]["probability"] == 0.0
    assert table[tuple(valid)]["probability"] == 1.0
    assert reason in caplog.text


def test_without_psl_uses_supplied_mask_without_renormalising(caplog):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    df = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 24], [55, 24]],
            "probability": [0.4, 0.6],
            "mask": [False, True],
            "cost": [123.0, 456.0],
        }
    )
    choice.execute_from_pandas(df)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[71, 24]["probability"] == 0.0
    assert table[71, 24]["cost"] == 123.0
    assert table[55, 24]["cost"] == 456.0
    assert table[55, 24]["probability"] == pytest.approx(0.6)
    assert not table[71, 24]["mask"]
    loads = choice.get_load_results()["flow_tot"]
    assert loads.loc[71] == 0.0
    assert loads.loc[24] == pytest.approx(0.6)
    assert "Invalid route" not in caplog.text


@pytest.mark.parametrize("routes", [[], [[]]])
@pytest.mark.parametrize("recompute_psl", [False, True])
def test_imported_empty_route_set_loads_no_demand(routes, recompute_psl):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    df = pd.DataFrame(
        {
            "origin id": [10] * len(routes),
            "destination id": [40] * len(routes),
            "route set": routes,
            "probability": [1.0] * len(routes),
        }
    )
    choice.execute_from_pandas(df, recompute_psl=recompute_psl)
    assert choice.get_results().empty
    assert np.all(choice.get_load_results()["flow_tot"] == 0.0)


@pytest.mark.parametrize("recompute_psl,expected_probability", [(False, 0.75), (True, 1.0)])
def test_empty_route_row_is_omitted_from_nonempty_set(recompute_psl, expected_probability):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    df = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[], [71, 12]],
            "probability": [0.25, 0.75],
        }
    )
    choice.execute_from_pandas(df, recompute_psl=recompute_psl)
    rows = choice.get_results()
    assert len(rows) == 1
    assert rows["route set"].iloc[0].tolist() == [71, 12]
    assert rows["probability"].iloc[0] == pytest.approx(expected_probability)
    assert choice.get_load_results()["flow_tot"].loc[71] == pytest.approx(expected_probability)


def test_path_file_without_psl_uses_supplied_mask(tmp_path, caplog):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    file = tmp_path / "invalid_route.parquet"
    pd.DataFrame(
        {
            "origin id": pd.Series([10], dtype="uint32"),
            "destination id": pd.Series([40], dtype="uint32"),
            "route set": [[71, 24]],
            "probability": [1.0],
            "mask": [False],
        }
    ).to_parquet(file, index=False)
    choice.execute_from_path_files(file)
    assert choice.get_results()["probability"].iloc[0] == 0.0
    assert choice.get_load_results()["flow_tot"].loc[71] == 0.0
    assert "disconnected links" not in caplog.text


def test_turn_penalties_are_costed_with_psl_without_demand():
    choice = RouteChoice(diamond())
    df = pd.DataFrame({"origin id": [10], "destination id": [40], "route set": [[71, 12]]})
    result = choice.recompute_psl(df)
    assert result["cost"].iloc[0] == 2.5
    assert result["mask"].iloc[0]


def test_route_choice_keeps_borrowed_graph_alive():
    graph = diamond()
    choice = RouteChoice(graph)
    del graph
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.execute_single(10, 40, demand=1.0)
    assert sorted(choice.get_results()["cost"]) == [2.5, 4.0]
    assert {tuple(route) for route in choice.get_results()["route set"]} == {(71, 12), (55, 24)}
    supplied = pd.DataFrame({"origin id": [10], "destination id": [40], "route set": [[71, 12]]})
    assert choice.recompute_psl(supplied)["cost"].iloc[0] == 2.5


def test_validation_preserves_rows_and_logs_each_reason(caplog):
    choice = RouteChoice(diamond(np.inf))
    supplied = pd.DataFrame(
        {
            "origin id": [20, 10, 10, 10],
            "destination id": [30, 40, 40, 40],
            "route set": [[71, 12], [71, 24], [55, 24], []],
            "cost": [999.0] * 4,
            "mask": [True, True, False, True],
        },
        index=[9, 9, 2, 1],
    )
    before = supplied.copy(deep=True)
    result = choice.recompute_psl(supplied)
    pd.testing.assert_frame_equal(supplied, before)
    assert result.index.tolist() == [9, 9, 2]
    assert result["mask"].tolist() == [False, False, False]
    assert not any(column.startswith("valid ") for column in result)
    assert np.isinf(result["cost"].iloc[:2]).all()
    assert result["cost"].iloc[2] == 4.0
    for reason in (
        "starts at the wrong node",
        "ends at the wrong node",
        "prohibited turn",
        "disconnected links",
        "Ignoring empty route",
    ):
        assert reason in caplog.text

    caplog.clear()
    pd.testing.assert_frame_equal(result, choice.recompute_psl(supplied, log_warnings=False))
    assert not caplog.records


def test_public_psl_discards_costs_and_preserves_exclusions():
    choice = RouteChoice(diamond(0.0))
    choice.set_choice_set_generation(cutoff_prob=1.0)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10, 10],
            "destination id": [40, 40, 40],
            "route set": [[71, 12], [55, 24], [55, 24]],
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


@pytest.mark.parametrize("recompute", [False, True])
def test_imported_supplied_mask_excludes_loading_and_select_links(recompute):
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.set_select_links({"banned": [[(71, 1)]], "used": [[(55, 1)]]})
    choice.add_demand(
        pd.DataFrame(
            {"flow": [10.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "mask": [False, True],
            "probability": [0.9, 0.1],
        }
    )
    choice.execute_from_pandas(supplied, recompute_psl=recompute)
    expected = 10.0 if recompute else 1.0
    loads = choice.get_load_results()["flow_tot"]
    assert loads.loc[71] == loads.loc[12] == 0.0
    assert loads.loc[55] == loads.loc[24] == pytest.approx(expected)
    selected = choice.get_select_link_loading_results()
    assert (selected["flow_banned_tot"] == 0.0).all()
    assert selected["flow_used_tot"].loc[55] == pytest.approx(expected)


def test_imported_routes_use_graph_prepared_before_route_choice():
    graph = diamond(5.0)
    graph.set_graph("distance")
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
        }
    )
    result = choice.recompute_psl(supplied)
    assert result["cost"].tolist() == [75.0, 110.0]
    assert result["path overlap"].tolist() == [1.0, 1.0]
    assert result["probability"].sum() == pytest.approx(1.0)


def compressed_chain():
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [11, 12, 13],
            "a_node": [10, 20, 30],
            "b_node": [20, 30, 40],
            "direction": [1, 1, 1],
            "time": [1.0, 2.0, 3.0],
        }
    )
    graph.prepare_graph(np.array([10, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    return graph


def test_validation_checks_links_before_compression():
    graph = compressed_chain()
    assert graph.compact_num_links == 1
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10] * 4,
            "destination id": [40] * 4,
            "route set": [[11, 12, 13], [11, 13], [12, 11, 13], [11, 12]],
        }
    )
    result = choice.recompute_psl(supplied, log_warnings=False)
    assert result["cost"].tolist() == [6.0, np.inf, np.inf, np.inf]
    assert result["mask"].tolist() == [True, False, False, False]
    assert result["probability"].tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("route", [[999], [-71]])
def test_missing_directed_links_are_errors_even_when_masked(route):
    choice = RouteChoice(diamond())
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [route],
            "mask": [False],
        }
    )
    with pytest.raises(ValueError, match="absent from the graph"):
        choice.recompute_psl(supplied, log_warnings=False)


def test_empty_routes_are_omitted_with_warning(caplog):
    choice = RouteChoice(diamond())
    supplied = pd.DataFrame({"origin id": [10], "destination id": [40], "route set": [[]]})
    result = choice.recompute_psl(supplied)
    assert result.empty
    assert result["mask"].dtype == bool
    assert result["cost"].dtype == np.float64
    assert "Ignoring empty route" in caplog.text


def test_infinite_link_cost_is_masked_without_a_turn_ban():
    graph = diamond()
    graph.graph.loc[graph.graph.link_id == 71, "time"] = np.inf
    graph.set_graph("time")
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
        }
    )
    result = choice.recompute_psl(supplied, log_warnings=False)
    assert np.isinf(result["cost"].iloc[0])
    assert result["mask"].tolist() == [False, True]
    assert result["probability"].tolist() == [0.0, 1.0]


def test_without_psl_does_not_validate_or_recost(caplog):
    choice = RouteChoice(diamond(np.inf))
    choice.set_choice_set_generation()
    choice.add_demand(
        pd.DataFrame(
            {"flow": [1.0]},
            index=pd.MultiIndex.from_tuples([(10, 40)], names=["origin id", "destination id"]),
        )
    )
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 24], [71, 12]],
            "cost": [123.0, 456.0],
            "probability": [0.4, 0.6],
        }
    )
    choice.execute_from_pandas(supplied)
    result = choice.get_results()
    assert result["mask"].tolist() == [True, True]
    assert result["cost"].tolist() == [123.0, 456.0]
    assert result["probability"].tolist() == [0.4, 0.6]
    assert not caplog.records


def test_recompute_psl_blocks_intermediate_centroid(caplog):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [11, 12, 13],
            "a_node": [10, 20, 10],
            "b_node": [20, 40, 40],
            "direction": [1, 1, 1],
            "time": [1.0, 1.0, 3.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(True)
    graph.set_graph("time")
    assert not graph.has_turn_restrictions  # Without explicit turns, block the centroid node.
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[11, 12], [13]],
        }
    )
    result = choice.recompute_psl(supplied)
    assert result["cost"].tolist() == [np.inf, 3.0]
    assert result["mask"].tolist() == [False, True]
    assert result["probability"].tolist() == [0.0, 1.0]
    assert "blocked centroid" in caplog.text or "prohibited turn" in caplog.text
