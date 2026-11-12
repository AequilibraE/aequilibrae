"""Route choice respects turn controls and keeps turn labels after each search is reused."""

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.cython.bridge import Bridge

from aequilibrae.matrix import from_df
from aequilibrae.paths import Network
from aequilibrae.paths.route_choice import RouteChoice


def junction(penalty, *, lonlat=None):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 10, 30, 20],
            "b_node": [20, 30, 20, 40],
            "direction": [1] * 4,
            "time": [1.0, 2.0, 1.0, 1.0],
        }
    )
    turns = (
        None
        if penalty is None
        else pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return Network(links, centroids=np.array([10, 20, 30, 40]), turn_restrictions=turns, lonlat=lonlat).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=False
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
@pytest.mark.parametrize("a_star", [False, True])
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
        reference = graph.compute_path(origin, destination)
        reference_ids, reference_directions = reference.get_network_links()
        np.testing.assert_array_equal(reference_ids, ids)
        np.testing.assert_array_equal(reference_directions, directions)
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


def single_od_demand(origin, destination, flow=1.0):
    return from_df(
        pd.DataFrame({"origin id": [origin], "destination id": [destination], "flow": [flow]}),
        "origin id",
        "destination id",
        sparse=True,
    )


def batched_junction_routes(graph, bfsle, max_routes):
    ods = [(10, 40), (20, 40)]
    frame = pd.DataFrame(
        {"flow": [1.0, 1.0]}, index=pd.MultiIndex.from_tuples(ods, names=["origin id", "destination id"])
    )
    demand = from_df(frame.reset_index(), "origin id", "destination id", index=graph.centroids, sparse=True)
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


def diamond(penalty=0.5, *, cost="time", lonlat=None):
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [30.0, 40.0, 50.0, 60.0],
        }
    )
    turns = (
        None
        if penalty is None
        else pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return Network(links, centroids=np.array([10, 20, 30, 40]), turn_restrictions=turns, lonlat=lonlat).build_graph(
        "c", cost=cost, compact=True, block_centroid_flows=False
    )


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_generated_routes_keep_original_cost_and_turns(algorithm):
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)

    # Search may penalise links, but PSL must use their original costs.
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
    routes = choice.execute_single(10, 40, demand=1.0)
    assert len(routes) == 1
    np.testing.assert_array_equal(routes[0][0], [55, 24])
    np.testing.assert_array_equal(routes[0][1], [1, 1])
    assert choice.get_results()["cost"].iloc[0] == pytest.approx(4.0)


@pytest.mark.parametrize("algorithm", ["bfsle", "link-penalisation"])
def test_generation_without_psl_does_not_need_turn_steps(algorithm):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation(algorithm, max_routes=2, max_depth=5, penalty=3.0)
    routes = choice.execute_single(10, 40)
    assert {(tuple(ids), tuple(directions)) for ids, directions in routes} == {
        ((71, 12), (1, 1)),
        ((55, 24), (1, 1)),
    }
    assert "cost" not in choice.get_results()


def recompute_imported_routes(graph, routes, *, destination=40, return_choice=False):
    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, destination))
    supplied = pd.DataFrame(
        {
            "origin id": [10] * len(routes),
            "destination id": [destination] * len(routes),
            "route set": routes,
            "route directions": [[1] * len(route) for route in routes],
        }
    )
    choice.execute_from_pandas(supplied, recompute_psl=True)
    return choice if return_choice else choice.get_results()


@pytest.mark.parametrize("penalty", [5.0, np.inf])
def test_recomputed_psl_uses_turn_penalties_and_rejects_bans(penalty):
    if not np.isfinite(penalty):
        with pytest.raises(ValueError, match="prohibited turn"):
            recompute_imported_routes(diamond(penalty), [[71, 12], [55, 24]])
        return

    choice = recompute_imported_routes(diamond(penalty), [[71, 12], [55, 24]], return_choice=True)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[55, 24]["cost"] == pytest.approx(4.0)
    assert table[71, 12]["cost"] == pytest.approx(2.0 + penalty)
    assert table[71, 12]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(penalty - 2.0)))


def test_recomputed_psl_all_routes_banned():
    with pytest.raises(ValueError, match="prohibited turn"):
        recompute_imported_routes(diamond(np.inf), [[71, 12]])


@pytest.mark.parametrize("allow_explicit_uturn", [False, True])
def test_recomputed_psl_applies_uturn_rules(allow_explicit_uturn):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 20, 10, 20],
            "b_node": [20, 10, 40, 40],
            "direction": [1] * 4,
            "time": [1.0] * 4,
        }
    )
    turns = pd.DataFrame(
        {
            "from_node": [10],
            "via_node": [20],
            "to_node": [10 if allow_explicit_uturn else 40],
            "penalty": [0.0],
        }
    )
    network = Network(links, centroids=np.array([10, 20, 40]), turn_restrictions=turns)
    graph = network.build_graph("c", cost="time", compact=True, block_centroid_flows=False)

    if allow_explicit_uturn:
        rows = recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
        table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
        assert table[11, 12, 13]["cost"] == pytest.approx(3.0)
        assert table[11, 12, 13]["mask"]
    else:
        with pytest.raises(ValueError, match="disallowed U-turn"):
            recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])

        graph = Network(links, centroids=np.array([10, 20, 40])).build_graph(
            "c", cost="time", compact=True, block_centroid_flows=False
        )
        node_rows = recompute_imported_routes(graph, [[11, 12, 13], [11, 14]])
        node_table = {tuple(row["route set"]): row for _, row in node_rows.iterrows()}
        # Node-based routing does not impose an additional U-turn policy on imports.
        assert node_table[11, 12, 13]["mask"]
        assert node_table[11, 14]["mask"]


def test_recomputed_psl_from_path_files_includes_turns(tmp_path):
    graph = diamond(5.0)
    source = RouteChoice(graph)
    source.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    source.execute_single(10, 40, demand=1.0)
    file = tmp_path / "routes.parquet"
    source.get_results().to_parquet(file, index=False)

    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    choice.execute_from_path_files(file, recompute_psl=True)
    rows = choice.get_results()
    table = {tuple(row["route set"]): row for _, row in rows.iterrows()}
    assert table[71, 12]["cost"] == pytest.approx(7.0)
    assert table[55, 24]["cost"] == pytest.approx(4.0)


@pytest.mark.parametrize("penalty", [None, 0.5])
def test_recomputed_psl_rejects_disconnected_links(penalty):
    with pytest.raises(ValueError, match="disconnected links"):
        recompute_imported_routes(diamond(penalty), [[71, 24], [55, 24]])


def test_imported_link_absent_from_compact_graph_is_rejected():
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 99],
            "a_node": [10, 20, 20],
            "b_node": [20, 40, 50],
            "direction": [0, 0, 0],
            "time": [1.0, 1.0, 1.0],
        }
    )
    graph = Network(links, centroids=np.array([10, 40])).build_graph(
        "c", cost="time", compact=True, remove_dead_ends=True, block_centroid_flows=False
    )
    choice = RouteChoiceSet(graph)
    demand = single_od_demand(10, 40)
    # The through route survives compression; only the spur to node 50 is removed.
    assert graph.mapping.network_link_to_graph[graph.mapping.network.link_index(99, 1)] == -1
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[71, 99, 99, 12]],
            "route directions": [[1, 1, -1, 1]],
        }
    )
    with pytest.raises(ValueError, match="absent from the compressed graph"):
        choice.assign_from_df(supplied, demand, select_links={}, recompute_psl=True)


@pytest.mark.parametrize(
    "origin,destination,invalid,valid,reason",
    [
        (20, 40, [55, 24], [12], "starts at the wrong node"),
        (10, 30, [71, 12], [55], "ends at the wrong node"),
    ],
)
@pytest.mark.parametrize("recompute", [False, True])
def test_imported_route_validation_checks_od_endpoints(origin, destination, invalid, valid, reason, recompute):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(origin, destination))
    df = pd.DataFrame(
        {
            "origin id": [origin, origin],
            "destination id": [destination, destination],
            "route set": [invalid, valid],
            "route directions": [[1] * len(invalid), [1] * len(valid)],
            "probability": [0.25, 0.75],
        }
    )
    with pytest.raises(ValueError, match=reason):
        choice.execute_from_pandas(df, recompute_psl=recompute)


def test_without_psl_uses_supplied_mask_without_renormalising():
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    df = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "route directions": [[1, 1], [1, 1]],
            "probability": [0.4, 0.6],
            "mask": [False, True],
            "cost": [123.0, 456.0],
        }
    )
    choice.execute_from_pandas(df)
    table = {tuple(row["route set"]): row for _, row in choice.get_results().iterrows()}
    assert table[71, 12]["probability"] == 0.0
    assert table[71, 12]["cost"] == 123.0
    assert table[55, 24]["cost"] == 456.0
    assert table[55, 24]["probability"] == pytest.approx(0.6)
    assert not table[71, 12]["mask"]
    loads = choice.get_load_results()["flow_tot"]
    assert loads.loc[71] == 0.0
    assert loads.loc[24] == pytest.approx(0.6)


@pytest.mark.parametrize("routes", [[], [[]]])
@pytest.mark.parametrize("recompute_psl", [False, True])
def test_imported_empty_route_set_loads_no_demand(routes, recompute_psl):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    df = pd.DataFrame(
        {
            "origin id": [10] * len(routes),
            "destination id": [40] * len(routes),
            "route set": routes,
            "route directions": [[1] * len(route) for route in routes],
            "probability": [1.0] * len(routes),
        }
    )
    if routes:
        with pytest.raises(ValueError, match="empty route"):
            choice.execute_from_pandas(df, recompute_psl=recompute_psl)
    else:
        choice.execute_from_pandas(df, recompute_psl=recompute_psl)
        assert choice.get_results().empty
        assert np.all(choice.get_load_results()["flow_tot"] == 0.0)


@pytest.mark.parametrize("recompute_psl", [False, True])
def test_empty_route_row_is_rejected_in_nonempty_set(recompute_psl):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    df = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[], [71, 12]],
            "route directions": [[], [1, 1]],
            "probability": [0.25, 0.75],
        }
    )
    with pytest.raises(ValueError, match="empty route"):
        choice.execute_from_pandas(df, recompute_psl=recompute_psl)


def test_path_file_without_psl_rejects_invalid_masked_routes(tmp_path):
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    file = tmp_path / "invalid_route.parquet"
    pd.DataFrame(
        {
            "origin id": pd.Series([10], dtype="uint64"),
            "destination id": pd.Series([40], dtype="uint64"),
            "route set": [[71, 24]],
            "route directions": [[1, 1]],
            "probability": [1.0],
            "mask": [False],
        }
    ).to_parquet(file, index=False)
    with pytest.raises(ValueError, match="disconnected links"):
        choice.execute_from_path_files(file)


def test_turn_penalties_are_costed_with_psl_without_demand():
    choice = RouteChoice(diamond())
    df = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": [[71, 12]], "route directions": [[1, 1]]}
    )
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
    supplied = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": [[71, 12]], "route directions": [[1, 1]]}
    )
    assert choice.recompute_psl(supplied)["cost"].iloc[0] == 2.5


@pytest.mark.parametrize(
    "origin,destination,route,reason",
    [
        (20, 40, [71, 12], "starts at the wrong node"),
        (10, 30, [71, 12], "ends at the wrong node"),
        (10, 40, [71, 12], "prohibited turn"),
        (10, 40, [71, 24], "disconnected links"),
        (10, 40, [], "empty route"),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_validation_rejects_routes_without_mutating_input(origin, destination, route, reason, masked, recompute):
    choice = RouteChoice(diamond(np.inf))
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(origin, destination))
    supplied = pd.DataFrame(
        {
            "origin id": [origin],
            "destination id": [destination],
            "route set": [route],
            "route directions": [[1] * len(route)],
            "cost": [999.0],
            "probability": [1.0],
            "mask": [not masked],
        },
        index=[9],
    )
    before = supplied.copy(deep=True)
    with pytest.raises(ValueError, match=f"row 9:.*{reason}"):
        choice.recompute_psl(supplied)
    with pytest.raises(ValueError, match=f"row 9:.*{reason}"):
        choice.execute_from_pandas(supplied, recompute_psl=recompute)
    pd.testing.assert_frame_equal(supplied, before)


def test_public_psl_discards_costs_and_preserves_exclusions():
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


@pytest.mark.parametrize("recompute", [False, True])
def test_imported_supplied_mask_excludes_loading_and_select_links(recompute):
    graph = diamond()
    choice = RouteChoice(graph)
    choice.set_choice_set_generation()
    choice.set_select_links({"banned": [[(71, 1)]], "used": [[(55, 1)]]})
    choice.set_demand(single_od_demand(10, 40, flow=10.0))
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "route directions": [[1, 1], [1, 1]],
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
    graph = diamond(5.0, cost="distance")
    choice = RouteChoice(graph)
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


def compressed_chain():
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13],
            "a_node": [10, 20, 30],
            "b_node": [20, 30, 40],
            "direction": [1, 1, 1],
            "time": [1.0, 2.0, 3.0],
        }
    )
    return Network(links, centroids=np.array([10, 40])).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=False
    )


def test_validation_checks_links_before_compression():
    graph = compressed_chain()
    assert graph.num_links == 1
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10] * 4,
            "destination id": [40] * 4,
            "route set": [[11, 12, 13], [11, 13], [12, 11, 13], [11, 12]],
            "route directions": [[1, 1, 1], [1, 1], [1, 1, 1], [1, 1]],
        }
    )
    with pytest.raises(ValueError, match="row 1:.*complete ordered graph-link expansions"):
        choice.recompute_psl(supplied)


@pytest.mark.parametrize("route", [[999], [-71]])
def test_missing_directed_links_are_errors_even_when_masked(route):
    choice = RouteChoice(diamond())
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [np.abs(route)],
            "route directions": [np.sign(route)],
            "mask": [False],
        }
    )
    with pytest.raises(ValueError, match="absent from the graph"):
        choice.recompute_psl(supplied)


def test_empty_routes_are_rejected():
    choice = RouteChoice(diamond())
    supplied = pd.DataFrame({"origin id": [10], "destination id": [40], "route set": [[]], "route directions": [[]]})
    with pytest.raises(ValueError, match="empty route"):
        choice.recompute_psl(supplied)


@pytest.mark.parametrize("recompute", [False, True])
def test_infinite_link_cost_is_rejected_without_a_turn_ban(recompute):
    graph = diamond()
    costs = np.ones(graph.mapping.network.link_count)
    costs[graph.mapping.network.link_ids == 71] = np.inf
    graph.set_costs(costs)
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[71, 12], [55, 24]],
            "route directions": [[1, 1], [1, 1]],
        }
    )
    with pytest.raises(ValueError, match="infinite route cost"):
        choice.recompute_psl(supplied)
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    supplied["probability"] = 0.5
    with pytest.raises(ValueError, match="infinite route cost"):
        choice.execute_from_pandas(supplied, recompute_psl=recompute)


def test_without_psl_validates_but_keeps_supplied_results():
    choice = RouteChoice(diamond())
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[55, 24], [71, 12]],
            "route directions": [[1, 1], [1, 1]],
            "cost": [123.0, 456.0],
            "probability": [0.4, 0.6],
        }
    )
    choice.execute_from_pandas(supplied)
    result = choice.get_results()
    assert result["mask"].tolist() == [True, True]
    assert result["cost"].tolist() == [123.0, 456.0]
    assert result["probability"].tolist() == [0.4, 0.6]


@pytest.mark.parametrize("recompute", [False, True])
def test_import_rejects_intermediate_centroid(recompute):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13],
            "a_node": [10, 20, 10],
            "b_node": [20, 40, 40],
            "direction": [1, 1, 1],
            "time": [1.0, 1.0, 3.0],
        }
    )
    graph = Network(links, centroids=np.array([10, 20, 40])).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=True
    )
    assert not graph.has_turn_restrictions  # Without explicit turns, block the centroid node.
    choice = RouteChoice(graph)
    supplied = pd.DataFrame(
        {
            "origin id": [10, 10],
            "destination id": [40, 40],
            "route set": [[11, 12], [13]],
            "route directions": [[1, 1], [1]],
        }
    )
    with pytest.raises(ValueError, match="blocked centroid|prohibited turn"):
        choice.recompute_psl(supplied)
    choice.set_choice_set_generation()
    choice.set_demand(single_od_demand(10, 40))
    supplied["probability"] = 0.5
    with pytest.raises(ValueError, match="blocked centroid|prohibited turn"):
        choice.execute_from_pandas(supplied, recompute_psl=recompute)
