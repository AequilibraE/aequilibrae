import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np
import pyarrow as pa

from aequilibrae import Graph, Project
from aequilibrae.paths.route_choice import RouteChoiceSet

from ...data import siouxfalls_project


# In these tests `max_depth` should be provided to prevent a runaway test case and just burning CI time
class TestRouteChoice(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_route_choice" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)

        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs(fields=["distance"], modes=["c"])
        self.graph = self.project.network.graphs["c"]  # type: Graph
        self.graph.set_graph("distance")
        self.graph.set_blocked_centroid_flows(False)

    def tearDown(self) -> None:
        self.project.close()

    def test_route_choice(self):
        rc = RouteChoiceSet(self.graph)
        a, b = 1, 20

        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                results = rc.run(a, b, max_routes=10, **kwargs)
                self.assertEqual(len(results), 10, "Returned more routes than expected")
                self.assertEqual(len(results), len(set(results)), "Returned duplicate routes")

                # With a depth 1 only one path will be found
                results = rc.run(a, b, max_routes=0, max_depth=1)
                self.assertEqual(len(results), 1, "Depth of 1 didn't yield a lone route")
                self.assertListEqual(
                    results, [(1, 5, 8, 12, 24, 29, 52, 58)], "Initial route isn't the shortest A* route"
                )

                # A depth of 2 should yield the same initial route plus the length of that route more routes minus duplicates and unreachable paths
                results2 = rc.run(a, b, max_routes=0, max_depth=2, **kwargs)
                self.assertTrue(results[0] in results2, "Initial route isn't present in a lower depth")

        self.assertListEqual(
            rc.run(a, b, max_routes=0, seed=0, max_depth=2),
            rc.run(a, b, max_routes=0, seed=10, max_depth=2),
            "Seeded and unseeded results differ with unlimited `max_routes` (queue is incorrectly being shuffled)",
        )

        self.assertNotEqual(
            rc.run(a, b, max_routes=3, seed=0, max_depth=2),
            rc.run(a, b, max_routes=3, seed=10, max_depth=2),
            "Seeded and unseeded results don't differ with limited `max_routes` (queue is not being shuffled)",
        )

    def test_route_choice_empty_path(self):
        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                rc = RouteChoiceSet(self.graph)
                a = 1

                self.assertFalse(
                    rc.batched([(a, a)], max_routes=0, max_depth=3, **kwargs),
                    "Route set from self to self should be empty",
                )

    def test_route_choice_blocking_centroids(self):
        for kwargs in [{"bfsle": True}, {"bfsle": False, "penalty": 1.1}]:
            with self.subTest(**kwargs):
                a, b = 1, 20

                self.graph.set_blocked_centroid_flows(False)
                rc = RouteChoiceSet(self.graph)

                results = rc.run(a, b, max_routes=2, max_depth=2, **kwargs)
                self.assertNotEqual(results, [], "Unblocked centroid flow found no paths")

                self.graph.set_blocked_centroid_flows(True)
                rc = RouteChoiceSet(self.graph)

                results = rc.run(a, b, max_routes=2, max_depth=2, **kwargs)
                self.assertListEqual(results, [], "Blocked centroid flow found a path")

    def test_route_choice_batched(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]

        max_routes = 20
        results = rc.batched(nodes, max_routes=max_routes, max_depth=10)

        gb = results.to_pandas().groupby(by="origin id")
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

        max_routes = 20
        with self.assertWarns(UserWarning):
            results = rc.batched(nodes, max_routes=max_routes, max_depth=10)

        gb = results.to_pandas().groupby(by="origin id")
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
                    rc.run(a, b, max_routes=max_routes, max_depth=max_depth)

        with self.assertRaises(ValueError):
            rc.run(1, 1, max_routes=1, max_depth=1, bfsle=True, penalty=1.5)
            rc.run(1, 1, max_routes=1, max_depth=1, bfsle=False, penalty=0.1)

    def test_round_trip(self):
        np.random.seed(1000)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]

        max_routes = 20

        path = join(self.project.project_base_path, "batched results")
        table = rc.batched(nodes, max_routes=max_routes, max_depth=10, cores=1)
        rc.batched(nodes, max_routes=max_routes, max_depth=10, cores=1, where=path)

        dataset = pa.dataset.dataset(path, format="parquet", partitioning=pa.dataset.HivePartitioning(rc.schema))
        new_table = (
            dataset.to_table()
            .to_pandas()
            .sort_values(by=["origin id", "destination id"])[["origin id", "destination id", "route set"]]
            .reset_index(drop=True)
        )

        table = table.to_pandas().sort_values(by=["origin id", "destination id"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(table, new_table)

    def test_frequency_results(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        table, freqs = rc.batched(nodes, max_routes=20, max_depth=10, freq_as_well=True)
        table = table.to_pandas()

        gb = table.groupby(by=["origin id", "destination id"])
        for od, freq in zip(set(nodes), freqs):  # iteration order is changed by set operation
            df = gb.get_group(od)
            bincount = np.bincount(np.hstack(df["route set"].values))
            keys = bincount.nonzero()[0]

            vals = bincount[keys]
            self.assertListEqual(list(keys), freq[0], "Keys of frequencies differreturns")
            self.assertListEqual(list(vals), freq[1], "Values of frequencies differ")

    def test_cost_results(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        table, costs = rc.batched(nodes, max_routes=20, max_depth=10, cost_as_well=True)
        table = table.to_pandas()

        gb = table.groupby(by=["origin id", "destination id"])
        for od, cost_vec in zip(set(nodes), costs):  # iteration order is changed by set operation
            df = gb.get_group(od)
            for route, cost in zip(df["route set"].values, cost_vec):
                np.testing.assert_almost_equal(self.graph.cost[route].sum(), cost, err_msg=f"Cost differs for OD {od}")


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
