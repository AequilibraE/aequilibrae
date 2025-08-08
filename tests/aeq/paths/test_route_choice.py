import os
import pathlib
from os.path import join

import numpy as np
import pandas as pd
import pytest

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix, GeneralisedCOODemand
from aequilibrae.paths.route_choice import RouteChoice
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet


@pytest.fixture(scope="function")
def route_choice_setup(sioux_falls_single_class):
    sioux_falls_single_class.network.build_graphs(fields=["distance", "free_flow_time"], modes=["c"])
    graph = sioux_falls_single_class.network.graphs["c"]
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    mat = sioux_falls_single_class.matrices.get_matrix("demand_omx")
    mat.computational_view()

    yield {"project": sioux_falls_single_class, "graph": graph, "mat": mat, "rc": RouteChoice(graph)}

    mat.close()
    sioux_falls_single_class.close()


def test_prepare(route_choice_setup):
    rc = route_choice_setup["rc"]

    with pytest.raises(ValueError):
        rc.prepare([])

    with pytest.raises(ValueError):
        rc.prepare(["1", "2"])

    with pytest.raises(ValueError):
        rc.prepare([("1", "2")])

    with pytest.raises(ValueError):
        rc.prepare([1])

    rc.prepare([1, 2])
    assert list(rc.demand.df.index) == [(1, 2), (2, 1)]

    rc.prepare([(1, 2)])
    assert list(rc.demand.df.index) == [(1, 2)]


def test_set_save_routes(route_choice_setup):
    rc = route_choice_setup["rc"]

    with pytest.raises(ValueError):
        rc.set_save_routes("/non-existent-path")


def test_set_choice_set_generation(route_choice_setup):
    rc = route_choice_setup["rc"]

    rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
    assert rc.parameters == {
        "seed": 0,
        "max_routes": 20,
        "max_depth": 0,
        "max_misses": 100,
        "penalty": 1.1,
        "cutoff_prob": 0.0,
        "beta": 1.0,
        "store_results": True,
    }

    rc.set_choice_set_generation("bfsle", max_routes=20)
    assert rc.parameters == {
        "seed": 0,
        "max_routes": 20,
        "max_depth": 0,
        "max_misses": 100,
        "penalty": 1.0,
        "cutoff_prob": 0.0,
        "beta": 1.0,
        "store_results": True,
    }

    with pytest.raises(AttributeError):
        rc.set_choice_set_generation("not an algorithm", max_routes=20, penalty=1.1)


def test_link_results(route_choice_setup):
    rc = route_choice_setup["rc"]
    mat = route_choice_setup["mat"]

    rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
    rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})
    rc.add_demand(mat)
    rc.prepare()
    rc.execute(perform_assignment=True)

    df = rc.get_load_results()
    u_sl = rc.get_select_link_loading_results()

    assert list(df.columns) == [f"{mat_name}_{dir}" for dir in ["ab", "ba", "tot"] for mat_name in mat.names]

    assert list(u_sl.columns) == [
        f"{mat_name}_{sl_name}_{dir}" for sl_name in ["sl1", "sl2"] for dir in ["ab", "ba"] for mat_name in mat.names
    ] + [f"{mat_name}_{sl_name}_tot" for sl_name in ["sl1", "sl2"] for mat_name in mat.names]


@pytest.mark.parametrize("recompute_psl,change_cost", [(False, False), (True, False), (True, True)])
def test_execute_from_path_files(route_choice_setup, recompute_psl, change_cost):
    project = route_choice_setup["project"]
    graph = route_choice_setup["graph"]
    mat = route_choice_setup["mat"]
    rc = route_choice_setup["rc"]

    tmp_path = pathlib.Path(project.project_base_path) / "rc"
    tmp_path.mkdir()

    rc.set_choice_set_generation("link-penalization", max_routes=5, penalty=1.1)
    rc.add_demand(mat)
    rc.prepare()
    rc.execute(perform_assignment=True)

    results = rc.get_results()
    ll_res = rc.get_load_results()
    rc.save_path_files(tmp_path)

    if change_cost:
        graph.set_graph("free_flow_time")

    rc2 = RouteChoice(graph)
    rc2.add_demand(mat)
    rc2.set_choice_set_generation(store_results=True)

    rc2.execute_from_path_files(tmp_path, recompute_psl=recompute_psl)
    results_new = rc2.get_results()
    ll_res_new = rc2.get_load_results()

    if recompute_psl and change_cost:
        # Recomputing PSL and changing the cost field means the link loads will be different (cost change), and path
        # overlap, cost, and mask will be different (recompute PSL)
        pd.testing.assert_frame_equal(
            results[["origin id", "destination id", "route set"]],
            results_new[["origin id", "destination id", "route set"]],
        )
    elif recompute_psl and not change_cost:
        # Everything must match here
        pd.testing.assert_frame_equal(results, results_new)
        pd.testing.assert_frame_equal(ll_res, ll_res_new)
    elif not recompute_psl and change_cost:
        raise RuntimeError("branch should be unreachable, cannot change cost field without recomputing PSL")
    elif not recompute_psl and not change_cost:
        # Cost, mask, and path overlap are not compared because the they are note used when assigning from DF
        pd.testing.assert_frame_equal(
            results[["origin id", "destination id", "route set", "probability"]],
            results_new[["origin id", "destination id", "route set", "probability"]],
        )
        # Not recomputing PSL means the assignment results are the same
        pd.testing.assert_frame_equal(ll_res, ll_res_new)
    else:
        raise  # Unreachable


def test_saving(route_choice_setup):
    project = route_choice_setup["project"]
    mat = route_choice_setup["mat"]
    rc = route_choice_setup["rc"]

    rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
    rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})
    rc.add_demand(mat)
    rc.prepare()
    rc.execute(perform_assignment=True)
    lloads = rc.get_load_results()
    u_sl = rc.get_select_link_loading_results()

    rc.save_link_flows("ll")
    rc.save_select_link_flows("sl")

    with project.results_connection as conn:
        for table, df in [
            ("ll_uncompressed", lloads),
            ("sl_uncompressed", u_sl),
        ]:
            df2 = pd.read_sql(f"select * from {table}", conn).set_index("link_id")
            df3 = project.results.get_results(table).set_index("link_id")

            pd.testing.assert_frame_equal(df2, df)
            pd.testing.assert_frame_equal(df3, df)

    matrices = AequilibraeMatrix()
    matrices.create_from_omx((pathlib.Path(project.project_base_path) / "matrices" / "sl").with_suffix(".omx"))
    matrices.computational_view()

    for sl_name, v in rc.get_select_link_od_matrix_results().items():
        for demand_name, matrix in v.items():
            np.testing.assert_allclose(matrix.to_scipy().toarray(), matrices.matrix[sl_name + "_" + demand_name])

    matrices.close()


def test_round_trip(route_choice_setup):
    project = route_choice_setup["project"]
    mat = route_choice_setup["mat"]
    rc = route_choice_setup["rc"]

    rc.add_demand(mat)
    rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
    rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})
    rc.prepare()

    path = join(project.project_base_path, "batched results")
    os.mkdir(path)

    rc.set_save_routes(None)
    rc.execute(perform_assignment=True)
    table = rc.get_results()

    rc.set_save_routes(path)
    rc.execute(perform_assignment=True)
    table2 = rc.get_results()

    table = table.sort_values(by=["origin id", "destination id", "cost"]).reset_index(drop=True)
    table2 = table2[table.columns].sort_values(by=["origin id", "destination id", "cost"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(table, table2)


def test_assign_from_df(route_choice_setup):
    graph = route_choice_setup["graph"]
    rc = RouteChoiceSet(graph)

    mat2 = AequilibraeMatrix()
    mat2.create_empty(
        memory_only=True,
        zones=graph.num_zones,
        matrix_names=["all ones"],
    )
    mat2.index = graph.centroids[:]
    mat2.computational_view()
    mat2.matrix_view[:, :] = np.full((graph.num_zones, graph.num_zones), 1.0)
    demand = GeneralisedCOODemand(
        "origin id",
        "destination id",
        graph.nodes_to_indices,
        shape=(graph.num_zones, graph.num_zones),
    )
    demand.add_matrix(mat2)

    args = {
        "graph": graph.graph,
        "demand": demand,
        "select_links": {},
        "recompute_psl": False,
        "sl_link_loading": False,
        "store_results": True,
    }

    # Test missing OD pairs
    with pytest.raises(KeyError):
        df = pd.DataFrame(
            {
                "origin id": [graph.centroids[0]],
                "destination id": [999999999],
                "probability": [1.0],
            }
        )
        rc.assign_from_df(df=df, **args)

    # Test link missing from compressed graph
    with pytest.raises(KeyError):
        df = pd.DataFrame(
            {
                "origin id": [graph.centroids[0]],
                "destination id": [graph.centroids[1]],
                "probability": [1.0],
                "route set": [[999999999]],
            }
        )
        rc.assign_from_df(df=df, **args)

    # Test wrong direction
    with pytest.raises(KeyError):
        df = pd.DataFrame(
            {
                "origin id": [graph.centroids[0]],
                "destination id": [graph.centroids[1]],
                "probability": [1.0],
                "route set": [[-graph.graph.iloc[0].link_id]],
            }
        )
        rc.assign_from_df(df=df, **args)

    # Test route sets should be a list
    with pytest.raises(TypeError):
        df = pd.DataFrame(
            {
                "origin id": [graph.centroids[0]],
                "destination id": [graph.centroids[1]],
                "probability": [1.0],
                "route set": [1],
            }
        )
        rc.assign_from_df(df=df, **args)

    # Tests that should not raise errors
    link_ids = graph.network.sample(10).link_id.to_numpy()
    links = graph.network[graph.network.link_id.isin(link_ids)]
    links = links.loc[links.link_id.replace({x: i for i, x in enumerate(link_ids)}).sort_values().index]

    values = np.random.random(len(links))
    df = pd.DataFrame(
        {
            "origin id": links.a_node.to_numpy(),
            "destination id": links.b_node.to_numpy(),
            "probability": values,
            "route set": [[x] for x in link_ids],
        }
    )

    # Reduce demand matrix to half the route sets
    args["demand"].df = args["demand"].df.loc[
        df[["origin id", "destination id"]].head(len(df) // 2).itertuples(name=None, index=False)
    ]

    # Test with recompute_psl=False
    rc.assign_from_df(df=df, **args)
    results = rc.get_results()
    ll_res = rc.get_link_loading()["all ones"]

    values2 = np.zeros_like(values)
    values2[: len(df) // 2] = values[: len(df) // 2]
    np.testing.assert_array_equal(
        ll_res[links.index],
        values2,
        "Link loading results don't match the expected values",
    )

    np.testing.assert_array_equal(results["cost"].to_numpy(), 0.0)
    np.testing.assert_array_equal(results["mask"].to_numpy(), True)
    np.testing.assert_array_equal(results["path overlap"].to_numpy(), 0.0)

    pd.testing.assert_series_equal(
        results["probability"],
        df.head(len(df) // 2)["probability"],
        check_index=False,
    )

    # Test with recompute_psl=True
    rc.assign_from_df(df=df, **(args | {"recompute_psl": True}))
    results = rc.get_results()
    ll_res = rc.get_link_loading()["all ones"]

    values2 = np.zeros(len(links.index))
    values2[: len(df) // 2] = 1.0
    np.testing.assert_array_equal(
        ll_res[links.index],
        values2,
        "Link loading results don't match the expected values",
    )

    links2 = links.head(len(df) // 2)
    np.testing.assert_array_equal(results["cost"].to_numpy(), links2["distance"].to_numpy())
    np.testing.assert_array_equal(results["mask"].to_numpy(), True)
    np.testing.assert_array_equal(results["path overlap"].to_numpy(), 1.0)
    np.testing.assert_array_equal(results["probability"].to_numpy(), 1.0)


@pytest.mark.parametrize("cost", ["distance", "free_flow_time"])
def test_known_results(route_choice_setup, cost):
    graph = route_choice_setup["graph"]
    rc = RouteChoiceSet(graph)

    graph.set_graph(cost)

    np.random.seed(0)
    nodes = [tuple(x) for x in np.random.choice(graph.centroids, size=(10, 2), replace=False)]

    mat = AequilibraeMatrix()
    mat.create_empty(
        memory_only=True,
        zones=graph.num_zones,
        matrix_names=["all zeros", "single one"],
    )
    mat.index = graph.centroids[:]
    mat.computational_view()
    mat.matrix_view[:, :, 0] = np.full((graph.num_zones, graph.num_zones), 1.0)
    mat.matrix_view[:, :, 1] = np.zeros((graph.num_zones, graph.num_zones))
    demand = GeneralisedCOODemand(
        "origin id",
        "destination id",
        graph.nodes_to_indices,
        shape=(graph.num_zones, graph.num_zones),
    )
    demand.add_matrix(mat)

    demand.df.loc[nodes] = 0.0
    demand.df.loc[nodes[0], "single one"] = 1.0
    demand.df = demand.df.loc[nodes].fillna(0.0)

    rc.batched(demand, max_routes=20, max_depth=10, path_size_logit=True)

    link_loads = rc.get_link_loading()
    table = rc.get_results()

    # Test all zeros matrix
    u = link_loads["all zeros"]
    np.testing.assert_allclose(u, 0.0)

    # Test single one matrix
    u = link_loads["single one"]
    link = graph.graph[(graph.graph.a_node == nodes[0][0] - 1) & (graph.graph.b_node == nodes[0][1] - 1)]

    lid = link.link_id.values[0]
    t = table[table["route set"].apply(lambda x, lid=lid: lid in set(x))]
    v = t.probability.sum()

    assert abs(u[lid - 1] - v) < 1e-6


@pytest.mark.parametrize("cost", ["distance", "free_flow_time"])
def test_select_link(route_choice_setup, cost):
    graph = route_choice_setup["graph"]
    rc = RouteChoiceSet(graph)

    graph.set_graph(cost)

    np.random.seed(0)
    nodes = [tuple(x) for x in np.random.choice(graph.centroids, size=(10, 2), replace=False)]
    demand = demand_from_nodes(nodes, graph)

    mat = AequilibraeMatrix()
    mat.create_empty(
        memory_only=True,
        zones=graph.num_zones,
        matrix_names=["all ones"],
    )
    mat.index = graph.centroids[:]
    mat.computational_view()
    mat.matrix_view[:, :] = np.full((graph.num_zones, graph.num_zones), 1.0)
    demand.add_matrix(mat)
    demand.df = demand.df.loc[nodes]

    rc.batched(
        demand,
        {
            "sl1": frozenset(frozenset((x,)) for x in graph.graph.set_index("link_id").loc[[23, 26]].__compressed_id__),
            "sl2": frozenset(frozenset((x,)) for x in graph.graph.set_index("link_id").loc[[11]].__compressed_id__),
        },
        max_routes=20,
        max_depth=10,
        path_size_logit=True,
    )
    table = rc.get_results()

    # Shortest routes between 20-4, and 21-2 share links 23 and 26. Link 26 also appears in between 10-8 and
    # 17-9 20-4 also shares 11 with 5-3
    ods = [(20, 4), (21, 2), (10, 8), (17, 9)]
    sl_link_loads = rc.get_sl_link_loading()
    sl_od_matrices = rc.get_sl_od_matrices()

    m = sl_od_matrices["sl1"]["all ones"].to_scipy()
    m2 = sl_od_matrices["sl2"]["all ones"].to_scipy()
    assert set(zip(*(m.toarray() > 0.0001).nonzero())) == {(o - 1, d - 1) for o, d in ods}
    assert set(zip(*(m2.toarray() > 0.0001).nonzero())) == {(20 - 1, 4 - 1), (5 - 1, 3 - 1)}

    u = sl_link_loads["sl1"]["all ones"]
    u2 = sl_link_loads["sl2"]["all ones"]

    t1 = table[(table.probability > 0.0) & table["route set"].apply(lambda x: bool(set(x) & {23, 26}))]
    t2 = table[(table.probability > 0.0) & table["route set"].apply(lambda x: 11 in set(x))]
    sl1_link_union = np.unique(np.hstack(t1["route set"].values))
    sl2_link_union = np.unique(np.hstack(t2["route set"].values))

    np.testing.assert_equal(u.nonzero()[0] + 1, sl1_link_union)
    np.testing.assert_equal(u2.nonzero()[0] + 1, sl2_link_union)

    assert abs(u.sum() - (t1["route set"].apply(len) * t1.probability).sum()) < 1e-6
    assert abs(u2.sum() - (t2["route set"].apply(len) * t2.probability).sum()) < 1e-6


# Helper function from original file
def demand_from_nodes(nodes, graph):
    demand = GeneralisedCOODemand(
        "origin id", "destination id", graph.nodes_to_indices, shape=(graph.num_zones, graph.num_zones)
    )
    df = pd.DataFrame()
    df.index = pd.MultiIndex.from_tuples(nodes, names=["origin id", "destination id"])
    demand.add_df(df)
    return demand
