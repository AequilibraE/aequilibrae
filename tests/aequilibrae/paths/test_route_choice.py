import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np

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

        results = rc.run(a, b, max_routes=10, max_depth=0)
        self.assertEqual(len(results), 10, "Returned more routes than expected")
        self.assertEqual(len(results), len(set(results)), "Returned duplicate routes")

        # With a depth 1 only one path will be found
        results = rc.run(a, b, max_routes=0, max_depth=1)
        self.assertEqual(len(results), 1, "Depth of 1 didn't yield a lone route")
        self.assertListEqual(results, [(58, 52, 29, 24, 12, 8, 5, 1)], "Initial route isn't the shortest A* route")

        # A depth of 2 should yield the same initial route plus the length of that route more routes minus duplicates and unreachable paths
        results2 = rc.run(a, b, max_routes=0, max_depth=2)
        self.assertEqual(
            len(results2), 1 + len(results[0]) - 4, "Depth of 2 didn't yield the expected number of routes"
        )
        self.assertTrue(results[0] in results2, "Initial route isn't present in a lower depth")

        self.assertListEqual(
            rc.run(a, b, max_routes=0, max_depth=2, seed=0),
            rc.run(a, b, max_routes=0, max_depth=2, seed=10),
            "Seeded and unseeded results differ with unlimited `max_routes` (queue is incorrectly being shuffled)",
        )

        self.assertNotEqual(
            rc.run(a, b, max_routes=3, max_depth=2, seed=0),
            rc.run(a, b, max_routes=3, max_depth=2, seed=10),
            "Seeded and unseeded results don't differ with limited `max_routes` (queue is not being shuffled)",
        )

    def test_route_choice_empty_path(self):
        rc = RouteChoiceSet(self.graph)
        a = 1

        self.assertEqual(
            rc.batched([(a, a)], max_routes=0, max_depth=3), {(a, a): []}, "Route set from self to self should be empty"
        )

    def test_route_choice_blocking_centroids(self):
        a, b = 1, 20

        self.graph.set_blocked_centroid_flows(False)
        rc = RouteChoiceSet(self.graph)

        results = rc.run(a, b, max_routes=2, max_depth=2)
        self.assertNotEqual(results, [], "Unblocked centroid flow found no paths")

        self.graph.set_blocked_centroid_flows(True)
        rc = RouteChoiceSet(self.graph)

        results = rc.run(a, b, max_routes=2, max_depth=2)
        self.assertListEqual(results, [], "Blocked centroid flow found a path")

    def test_route_choice_batched(self):
        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]

        max_routes = 20
        results = rc.batched(nodes, max_routes=max_routes, max_depth=10, cores=1)

        self.assertEqual(len(results), len(nodes), "Requested number of route sets not returned")

        for od, route_set in results.items():
            self.assertEqual(len(route_set), len(set(route_set)), f"Duplicate routes returned for {od}")
            self.assertEqual(len(route_set), max_routes, f"Requested number of routes not returned for {od}")

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

    def test_debug(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import matplotlib.pyplot as plt

        np.random.seed(0)
        rc = RouteChoiceSet(self.graph)
        # nodes = [tuple(x) for x in np.random.choice(self.graph.centroids, size=(10, 2), replace=False)]
        nodes = [(1, x) for x in self.graph.centroids if x != 1]

        max_routes = 20
        results: dict[
            tuple[int, int],  # OD keys
            list[  # routes, length = max_routes
                tuple[int, ...]  # individual paths, length variable
            ]
        ]

        self.graph.centroids

        # Table of origins
        results = rc.batched(nodes, max_routes=max_routes, max_depth=10, cores=1)
        batch = pa.RecordBatch.from_arrays([pa.array(v) for v in results.values()], names=[str(k[1]) for k in results.keys()])
        table = pa.Table.from_batches([batch] * 5)

        lengths = [len(r) for k, v in results.items() for r in v]
        # plt.hist(lengths)
        # plt.show(block=False)

        ty = pa.map_(
            pa.uint32(),  # Destination
            pa.list_(  # route set
                pa.list_(pa.uint32()),  # path
                # list_size=max_routes  # We could make this fixed length, not sure if it has benefits?
            )
        )

        # Route map contains a column (a list below) for each origin.
        # Each column is a map of (key, value) pairs where key is the destination and value is a list of paths
        # paths are lists of uint32s. We don't need int64 since they'll never contain negative (invalid) link IDs
        route_map = pa.array([
            [(k[1], v) for k, v in results.items()],  # Real example, TODO: make coercion easier
            [(k[1] * 100, v[::-1]) for k, v in results.items()]  # different one just for the sake of it
        ], type=ty)

        struct = pa.struct([
            pa.field("column index", pa.list_(pa.int32())),  # List origin IDs at index they appear in the above ^^^
            pa.field("route map", pa.list_(route_map.type), nullable=False),  # Not sure why this has to be a pa.list_(route_map.type) but just route_map.type doesn't work
        ])

        route_choice = pa.array([{
            "column index": [1, 100],  # Here we encode the origin ID to index map separately. I couldn't figure out how to nest MapArrays/I'm not sure its needed
            "routes map": route_map
        }], type=struct)  # Because the thing struct really stores lists of fields we could store multiple route choices in the same table

        table = pa.Table.from_pydict({'route choice': route_choice})  # Need to wrap the whole thing in a table to keep arrow happy
        pq.write_table(table, "table.parquet")  # Pretty sure with this structure writing a partitioned data set doesn't work

        # breakpoint()
        assert False


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
