import os
import pathlib
import sqlite3
import uuid
import zipfile
from os.path import dirname, join
from tempfile import gettempdir
from typing import List, Tuple
from unittest import TestCase, skip

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix, GeneralisedCOODemand, Sparse
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.paths.cython.route_choice_set_results import RouteChoiceSetResults
from aequilibrae.paths.route_choice import RouteChoice

from ...data import siouxfalls_project


# In these tests `max_depth` should be provided to prevent a runaway test case and just burning CI time
class TestRouteChoiceSet(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_route_choice" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)

        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs(fields=["distance", "free_flow_time"], modes=["c"])
        self.graph = self.project.network.graphs["c"]
        self.graph.set_graph("distance")
        self.graph.set_blocked_centroid_flows(False)

        self.mat = self.project.matrices.get_matrix("demand_omx")
        self.mat.computational_view()

        self.shape = (self.graph.num_zones, self.graph.num_zones)

        self.demand = GeneralisedCOODemand("origin id", "destination id", self.graph.nodes_to_indices, self.shape)
        self.demand.add_matrix(self.mat)

    def tearDown(self) -> None:
        self.mat.close()
        self.project.close()

    def test_route_choice(self):
        rc = RouteChoiceSet(self.graph)
        a, b = 1, 20

        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}, {"bfsle": True, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                results = rc.run(a, b, self.shape, max_routes=10, **kwargs)
                self.assertEqual(len(results), 10, "Returned incorrect number of routes")
                self.assertEqual(len(results), len(set(results)), "Returned duplicate routes")

                # With a depth 1 only one path will be found
                results = rc.run(a, b, self.shape, max_routes=0, max_depth=1)
                self.assertEqual(len(results), 1, "Depth of 1 didn't yield a lone route")
                self.assertListEqual(
                    results, [(2, 6, 9, 13, 25, 30, 53, 59)], "Initial route isn't the shortest A* route"
                )

                # A depth of 2 should yield the same initial route plus the length of that route more routes minus
                # duplicates and unreachable paths
                results2 = rc.run(a, b, self.shape, max_routes=0, max_depth=2, **kwargs)
                self.assertTrue(results[0] in results2, "Initial route isn't present in a lower depth")

        self.assertListEqual(
            rc.run(a, b, self.shape, max_routes=0, seed=0, max_depth=2),
            rc.run(a, b, self.shape, max_routes=0, seed=10, max_depth=2),
            "Seeded and unseeded results differ with unlimited `max_routes` (queue is incorrectly being shuffled)",
        )

        self.assertNotEqual(
            rc.run(a, b, self.shape, max_routes=3, seed=0, max_depth=2),
            rc.run(a, b, self.shape, max_routes=3, seed=10, max_depth=2),
            "Seeded and unseeded results don't differ with limited `max_routes` (queue is not being shuffled)",
        )

    def test_route_choice_empty_path(self):
        demand = demand_from_nodes([(1, 1)], self.graph)

        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                rc = RouteChoiceSet(self.graph)

                rc.batched(demand, max_routes=0, max_depth=3, **kwargs)
                self.assertTrue(
                    rc.get_results().empty,
                    "Route set from self to self should be empty",
                )

    def test_route_choice_blocking_centroids(self):
        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                a, b = 1, 20

                self.graph.set_blocked_centroid_flows(False)
                rc = RouteChoiceSet(self.graph)

                results = rc.run(a, b, self.shape, max_routes=2, max_depth=2, **kwargs)
                self.assertNotEqual(results, [], "Unblocked centroid flow found no paths")

                self.graph.set_blocked_centroid_flows(True)
                rc = RouteChoiceSet(self.graph)

                with self.assertWarns(UserWarning):
                    results = rc.run(a, b, self.shape, max_routes=2, max_depth=2, **kwargs)
                self.assertListEqual(results, [], "Blocked centroid flow found a path")

    def test_route_choice_batched(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]

        max_routes = 20
        demand = demand_from_nodes(nodes, self.graph)
        rc.batched(demand, max_routes=max_routes, max_depth=10, max_misses=200)
        results = rc.get_results()

        gb = results.groupby(by="origin id")
        self.assertEqual(len(gb), len(nodes), "Requested number of route sets not returned")

        for _, row in gb:
            self.assertFalse(any(row["route set"].duplicated()), f"Duplicate routes returned for {row['origin id']}")
            self.assertEqual(
                len(row["route set"]), max_routes, f"Requested number of routes not returned for {row['origin id']}"
            )

    def test_route_choice_duplicates_batched(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [(1, 20)] * 5
        demand = demand_from_nodes(nodes, self.graph)

        max_routes = 20
        rc.batched(demand, max_routes=max_routes, max_depth=10)
        results = rc.get_results()

        gb = results.groupby(by="origin id")
        self.assertEqual(len(gb), 1, "Duplicates not dropped")

    def test_route_choice_exceptions(self):
        rc = RouteChoiceSet(self.graph)
        args = [
            (1, 20, 0, 0),
            (1, 20, -1, 0),
            (1, 20, 0, -1),
            (0, 20, 1, 1),
            (1, 0, 1, 1),
        ]

        for a, b, max_routes, max_depth in args:
            with self.subTest(a=a, b=b, max_routes=max_routes, max_depth=max_depth):
                with self.assertRaises(ValueError):
                    rc.run(a, b, self.shape, max_routes=max_routes, max_depth=max_depth)

    def test_round_trip(self):
        np.random.seed(1000)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        demand = demand_from_nodes(nodes, self.graph)

        max_routes = 20

        path = join(self.project.project_base_path, "batched results")
        rc.batched(demand, max_routes=max_routes, max_depth=10, path_size_logit=True)
        table = rc.get_results()
        rc.batched(demand, max_routes=max_routes, max_depth=10, path_size_logit=True, where=path, cores=1)

        new_table = RouteChoiceSetResults.read_dataset(path)
        new_table = new_table.sort_values(by=["origin id", "destination id", "cost"])[table.columns].reset_index(
            drop=True
        )

        table = table.sort_values(by=["origin id", "destination id", "cost"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(table, new_table)

    def test_cost_results(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        demand = demand_from_nodes(nodes, self.graph)
        rc.batched(demand, max_routes=20, max_depth=10, path_size_logit=True)

        table = rc.get_results()

        gb = table.groupby(by=["origin id", "destination id"])
        for od, df in gb:
            for route, cost in zip(df["route set"].values, df["cost"].values):
                np.testing.assert_almost_equal(
                    self.graph.network.set_index("link_id").loc[route][self.graph.cost_field].sum(),
                    cost,
                    err_msg=f", cost differs for OD {od}",
                )

    def test_path_overlap_results(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        demand = demand_from_nodes(nodes, self.graph)
        rc.batched(demand, max_routes=20, max_depth=10, path_size_logit=True)
        table = rc.get_results()

        gb = table.groupby(by=["origin id", "destination id"])
        for od, df in gb:
            self.assertTrue(all((df["path overlap"] > 0) & (df["path overlap"] <= 1)))

    def test_prob_results(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        demand = demand_from_nodes(nodes, self.graph)

        for kwargs in [{"cutoff_prob": 0.0}, {"cutoff_prob": 0.5}, {"cutoff_prob": 1.0}]:
            with self.subTest(**kwargs):
                rc.batched(demand, max_routes=20, max_depth=10, path_size_logit=True, **kwargs)
                table = rc.get_results()

                gb = table.groupby(by=["origin id", "destination id"])
                for od, df in gb:
                    self.assertAlmostEqual(1.0, sum(df["probability"].values), msg=", probability not close to 1.0")

    def test_assign_from_df(self):
        mat = AequilibraeMatrix()
        mat.create_empty(
            memory_only=True,
            zones=self.graph.num_zones,
            matrix_names=["all ones"],
        )
        mat.index = self.graph.centroids[:]
        mat.computational_view()
        mat.matrix_view[:, :] = np.full(self.shape, 1.0)
        demand = GeneralisedCOODemand(
            "origin id",
            "destination id",
            self.graph.nodes_to_indices,
            shape=(self.graph.num_zones, self.graph.num_zones),
        )
        demand.add_matrix(mat)

        rc = RouteChoiceSet(self.graph)

        args = {
            "graph": self.graph.graph,
            "demand": demand,
            "select_links": {},
            "recompute_psl": False,
            "sl_link_loading": False,
            "store_results": True,
        }

        with self.subTest("failed to notice missing OD pairs"), self.assertRaises(KeyError):
            df = pd.DataFrame(
                {
                    "origin id": [self.graph.centroids[0]],
                    "destination id": [999999999],
                    "probability": [1.0],
                },
            )

            rc.assign_from_df(df=df, **args)

        with self.subTest("failed to notice link missing from compressed graph"), self.assertRaises(KeyError):
            df = pd.DataFrame(
                {
                    "origin id": [self.graph.centroids[0]],
                    "destination id": [self.graph.centroids[1]],
                    "probability": [1.0],
                    "route set": [[999999999]],
                },
            )

            rc.assign_from_df(df=df, **args)

        with self.subTest("failed to notice with wrong direction"), self.assertRaises(KeyError):
            df = pd.DataFrame(
                {
                    "origin id": [self.graph.centroids[0]],
                    "destination id": [self.graph.centroids[1]],
                    "probability": [1.0],
                    "route set": [[-self.graph.graph.iloc[0].link_id]],
                },
            )

            rc.assign_from_df(df=df, **args)

        with self.subTest("route sets should be a list"), self.assertRaises(TypeError):
            df = pd.DataFrame(
                {
                    "origin id": [self.graph.centroids[0]],
                    "destination id": [self.graph.centroids[1]],
                    "probability": [1.0],
                    "route set": [1],
                },
            )

            rc.assign_from_df(df=df, **args)

        link_ids = self.graph.network.sample(10).link_id.to_numpy()
        links = self.graph.network[self.graph.network.link_id.isin(link_ids)]
        links = links.loc[links.link_id.replace({x: i for i, x in enumerate(link_ids)}).sort_values().index]

        values = np.random.random(len(links))
        df = pd.DataFrame(
            {
                "origin id": links.a_node.to_numpy(),
                "destination id": links.b_node.to_numpy(),
                "probability": values,
                "route set": [[x] for x in link_ids],
            },
        )

        # For the tests that should not raise errors we reduce our demand matrix to half the route sets
        args["demand"].df = args["demand"].df.loc[
            df[["origin id", "destination id"]].head(len(df) // 2).itertuples(name=None, index=False)
        ]
        with self.subTest("assign random links", recompute_psl=False):
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

            # The filled values should be the constant here, we didn't recompute PSL.
            np.testing.assert_array_equal(results["cost"].to_numpy(), 0.0)
            np.testing.assert_array_equal(results["mask"].to_numpy(), True)
            np.testing.assert_array_equal(results["path overlap"].to_numpy(), 0.0)

            pd.testing.assert_series_equal(
                results["probability"],
                df.head(len(df) // 2)["probability"],
                check_index=False,
            )

        with self.subTest("assign random links", recompute_psl=True):
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

            # Since we recomputed PSL the cost field should be updated. As we sampled the links and there's only one
            # route per OD (and Sioux falls has no links with the same A and B nodes) there's no overlap, thus their are
            # all 1.0 for probabilities and path overlap.
            links2 = links.head(len(df) // 2)
            np.testing.assert_array_equal(results["cost"].to_numpy(), links2["distance"].to_numpy())
            np.testing.assert_array_equal(results["mask"].to_numpy(), True)
            np.testing.assert_array_equal(results["path overlap"].to_numpy(), 1.0)
            np.testing.assert_array_equal(results["probability"].to_numpy(), 1.0)

    def test_known_results(self):
        for cost in ["distance", "free_flow_time"]:
            with self.subTest(cost=cost):
                self.graph.set_graph(cost)

                np.random.seed(0)
                rc = RouteChoiceSet(self.graph)
                nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]

                mat = AequilibraeMatrix()
                mat.create_empty(
                    memory_only=True,
                    zones=self.graph.num_zones,
                    matrix_names=["all zeros", "single one"],
                )
                mat.index = self.graph.centroids[:]
                mat.computational_view()
                mat.matrix_view[:, :, 0] = np.full(self.shape, 1.0)
                mat.matrix_view[:, :, 1] = np.zeros((self.graph.num_zones, self.graph.num_zones))
                demand = GeneralisedCOODemand(
                    "origin id",
                    "destination id",
                    self.graph.nodes_to_indices,
                    shape=(self.graph.num_zones, self.graph.num_zones),
                )
                demand.add_matrix(mat)

                demand.df.loc[nodes] = 0.0
                demand.df.loc[nodes[0], "single one"] = 1.0
                demand.df = demand.df.loc[nodes].fillna(0.0)

                rc.batched(demand, max_routes=20, max_depth=10, path_size_logit=True)

                link_loads = rc.get_link_loading()
                table = rc.get_results()

                with self.subTest(matrix="all zeros"):
                    u = link_loads["all zeros"]
                    np.testing.assert_allclose(u, 0.0)

                with self.subTest(matrix="single one"):
                    u = link_loads["single one"]
                    link = self.graph.graph[
                        (self.graph.graph.a_node == nodes[0][0] - 1) & (self.graph.graph.b_node == nodes[0][1] - 1)
                    ]

                    lid = link.link_id.values[0]
                    t = table[table["route set"].apply(lambda x, lid=lid: lid in set(x))]
                    v = t.probability.sum()

                    self.assertAlmostEqual(u[lid - 1], v, places=6)

    def test_select_link(self):
        for cost in ["distance", "free_flow_time"]:
            with self.subTest(cost=cost):
                self.graph.set_graph(cost)

                np.random.seed(0)
                rc = RouteChoiceSet(self.graph)
                nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
                demand = demand_from_nodes(nodes, self.graph)

                mat = AequilibraeMatrix()
                mat.create_empty(
                    memory_only=True,
                    zones=self.graph.num_zones,
                    matrix_names=["all ones"],
                )
                mat.index = self.graph.centroids[:]
                mat.computational_view()
                mat.matrix_view[:, :] = np.full(self.shape, 1.0)
                demand.add_matrix(mat)
                demand.df = demand.df.loc[nodes]

                rc.batched(
                    demand,
                    {
                        "sl1": frozenset(
                            frozenset((x,))
                            for x in self.graph.graph.set_index("link_id").loc[[23, 26]].__compressed_id__
                        ),
                        "sl2": frozenset(
                            frozenset((x,)) for x in self.graph.graph.set_index("link_id").loc[[11]].__compressed_id__
                        ),
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
                self.assertSetEqual(set(zip(*(m.toarray() > 0.0001).nonzero())), {(o - 1, d - 1) for o, d in ods})
                self.assertSetEqual(set(zip(*(m2.toarray() > 0.0001).nonzero())), {(20 - 1, 4 - 1), (5 - 1, 3 - 1)})

                u = sl_link_loads["sl1"]["all ones"]
                u2 = sl_link_loads["sl2"]["all ones"]

                t1 = table[(table.probability > 0.0) & table["route set"].apply(lambda x: bool(set(x) & {23, 26}))]
                t2 = table[(table.probability > 0.0) & table["route set"].apply(lambda x: 11 in set(x))]
                sl1_link_union = np.unique(np.hstack(t1["route set"].values))
                sl2_link_union = np.unique(np.hstack(t2["route set"].values))

                np.testing.assert_equal(u.nonzero()[0] + 1, sl1_link_union)
                np.testing.assert_equal(u2.nonzero()[0] + 1, sl2_link_union)

                self.assertAlmostEqual(u.sum(), (t1["route set"].apply(len) * t1.probability).sum())
                self.assertAlmostEqual(u2.sum(), (t2["route set"].apply(len) * t2.probability).sum())


class TestRouteChoice(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_route_choice" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)

        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs(fields=["distance", "free_flow_time"], modes=["c"])
        self.graph = self.project.network.graphs["c"]
        self.graph.set_graph("distance")
        self.graph.set_blocked_centroid_flows(False)

        self.mat = self.project.matrices.get_matrix("demand_omx")
        self.mat.computational_view()

        self.rc = RouteChoice(self.graph)

    def tearDown(self) -> None:
        self.mat.close()
        self.project.close()

    def test_prepare(self):
        with self.assertRaises(ValueError):
            self.rc.prepare([])

        with self.assertRaises(ValueError):
            self.rc.prepare(["1", "2"])

        with self.assertRaises(ValueError):
            self.rc.prepare([("1", "2")])

        with self.assertRaises(ValueError):
            self.rc.prepare([1])

        self.rc.prepare([1, 2])
        self.assertListEqual(list(self.rc.demand.df.index), [(1, 2), (2, 1)])
        self.rc.prepare([(1, 2)])
        self.assertListEqual(list(self.rc.demand.df.index), [(1, 2)])

    def test_set_save_routes(self):
        self.rc = RouteChoice(self.graph)

        with self.assertRaises(ValueError):
            self.rc.set_save_routes("/non-existent-path")

    def test_set_choice_set_generation(self):
        self.rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
        self.assertDictEqual(
            self.rc.parameters,
            {
                "seed": 0,
                "max_routes": 20,
                "max_depth": 0,
                "max_misses": 100,
                "penalty": 1.1,
                "cutoff_prob": 0.0,
                "beta": 1.0,
                "store_results": True,
            },
        )

        self.rc.set_choice_set_generation("bfsle", max_routes=20)
        self.assertDictEqual(
            self.rc.parameters,
            {
                "seed": 0,
                "max_routes": 20,
                "max_depth": 0,
                "max_misses": 100,
                "penalty": 1.0,
                "cutoff_prob": 0.0,
                "beta": 1.0,
                "store_results": True,
            },
        )

        with self.assertRaises(AttributeError):
            self.rc.set_choice_set_generation("not an algorithm", max_routes=20, penalty=1.1)

    def test_link_results(self):
        self.rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)

        self.rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})

        self.rc.add_demand(self.mat)
        self.rc.prepare()

        self.rc.execute(perform_assignment=True)

        df = self.rc.get_load_results()
        u_sl = self.rc.get_select_link_loading_results()

        pd.testing.assert_index_equal(
            df.columns,
            pd.MultiIndex.from_tuples([(mat_name, dir) for dir in ["ab", "ba", "tot"] for mat_name in self.mat.names]),
        )

        pd.testing.assert_index_equal(
            u_sl.columns,
            pd.MultiIndex.from_tuples(
                [
                    (mat_name, sl_name, dir)
                    for sl_name in ["sl1", "sl2"]
                    for dir in ["ab", "ba"]
                    for mat_name in self.mat.names
                ]
                + [(mat_name, sl_name, "tot") for sl_name in ["sl1", "sl2"] for mat_name in self.mat.names]
            ),
        )

    def test_execute_from_path_files(self):
        tmp_path = pathlib.Path(self.project.project_base_path) / "rc"
        tmp_path.mkdir()

        self.rc.set_choice_set_generation("link-penalization", max_routes=5, penalty=1.1)
        self.rc.add_demand(self.mat)
        self.rc.prepare()
        self.rc.execute(perform_assignment=True)

        results = self.rc.get_results()
        ll_res = self.rc.get_load_results()
        self.rc.save_path_files(tmp_path)

        rc = RouteChoice(self.graph)
        rc.add_demand(self.mat)
        rc.set_choice_set_generation(store_results=True)

        with self.subTest(recompute_psl=False):
            rc.execute_from_path_files(tmp_path, recompute_psl=False)
            results_new = rc.get_results()
            ll_res_new = rc.get_load_results()
            pd.testing.assert_frame_equal(ll_res, ll_res_new)
            pd.testing.assert_frame_equal(
                results[["origin id", "destination id", "route set", "probability"]],
                results_new[["origin id", "destination id", "route set", "probability"]],
            )

        with self.subTest(recompute_psl=True):
            rc.execute_from_path_files(tmp_path, recompute_psl=True)
            results_new = rc.get_results()
            ll_res_new = rc.get_load_results()
            pd.testing.assert_frame_equal(ll_res, ll_res_new)
            pd.testing.assert_frame_equal(results, results_new)

        with self.subTest(recompute_psl=True, msg="change the cost field"):
            self.graph.set_graph("free_flow_time")
            rc = RouteChoice(self.graph)
            rc.add_demand(self.mat)
            rc.set_choice_set_generation(store_results=True)

            rc.execute_from_path_files(tmp_path, recompute_psl=True)
            results_new = rc.get_results()
            ll_res_new = rc.get_load_results()

            pd.testing.assert_series_equal(results["origin id"], results["origin id"])
            pd.testing.assert_series_equal(results["destination id"], results["destination id"])
            pd.testing.assert_series_equal(results["route set"], results["route set"])
            pd.testing.assert_series_equal(results["path overlap"], results["path overlap"])
            # All other fields may be different

    def test_saving(self):
        self.rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
        self.rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})
        self.rc.add_demand(self.mat)
        self.rc.prepare()
        self.rc.execute(perform_assignment=True)
        lloads = self.rc.get_load_results()
        u_sl = self.rc.get_select_link_loading_results()

        self.rc.save_link_flows("ll")
        self.rc.save_select_link_flows("sl")

        conn = sqlite3.connect(pathlib.Path(self.project.project_base_path) / "results_database.sqlite")
        with conn:
            for table, df in [
                ("ll_uncompressed", lloads),
                ("sl_uncompressed", u_sl),
            ]:
                with self.subTest(table=table):
                    df2 = pd.read_sql(f"select * from {table}", conn).set_index("link_id")
                    # NOTE: Pandas to_sql serialises the columns of a multiindex as a str, to avoid annoying parsing we
                    # use eval here.
                    df2.columns = pd.MultiIndex.from_tuples([eval(x) for x in df2.columns])
                    pd.testing.assert_frame_equal(df2, df)
        conn.close()

        matrices = AequilibraeMatrix()
        matrices.create_from_omx((pathlib.Path(self.project.project_base_path) / "matrices" / "sl").with_suffix(".omx"))
        matrices.computational_view()

        for sl_name, v in self.rc.get_select_link_od_matrix_results().items():
            for demand_name, matrix in v.items():
                np.testing.assert_allclose(matrix.to_scipy().toarray(), matrices.matrix[sl_name + "_" + demand_name])

        matrices.close()

    def test_round_trip(self):
        self.rc.add_demand(self.mat)
        self.rc.set_choice_set_generation("link-penalization", max_routes=20, penalty=1.1)
        self.rc.set_select_links({"sl1": [((23, 1),), ((26, 1),)], "sl2": [((11, 1),)]})
        self.rc.prepare()

        path = join(self.project.project_base_path, "batched results")
        os.mkdir(path)

        self.rc.set_save_routes(None)
        self.rc.execute(perform_assignment=True)
        table = self.rc.get_results()

        self.rc.set_save_routes(path)
        self.rc.execute(perform_assignment=True)
        table2 = self.rc.get_results()

        table = table.sort_values(by=["origin id", "destination id", "cost"]).reset_index(drop=True)
        table2 = table2[table.columns].sort_values(by=["origin id", "destination id", "cost"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(table, table2)


def generate_line_strings(project, graph, results):
    """Debug method"""
    import geopandas as gpd
    import shapely

    links = project.network.links.data.set_index("link_id")
    df = []
    for od, route_set in results.items():
        for route in route_set:
            df.append(
                (
                    *od,
                    shapely.MultiLineString(
                        links.loc[graph.graph[graph.graph.__compressed_id__.isin(route)].link_id].geometry.to_list()
                    ),
                )
            )

    df = gpd.GeoDataFrame(df, columns=["origin", "destination", "geometry"])
    df.set_geometry("geometry")
    return df


def demand_from_nodes(nodes: List[Tuple[int, int]], graph):
    demand = GeneralisedCOODemand(
        "origin id", "destination id", graph.nodes_to_indices, shape=(graph.num_zones, graph.num_zones)
    )
    df = pd.DataFrame()
    df.index = pd.MultiIndex.from_tuples(nodes, names=["origin id", "destination id"])
    demand.add_df(df)
    return demand
