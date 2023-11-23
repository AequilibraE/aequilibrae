import os
import sys
import uuid
import zipfile
from os.path import join
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from itertools import product

import numpy as np

from aequilibrae import Project
from aequilibrae.paths import path_computation, Graph
from aequilibrae.paths.results import PathResults
from aequilibrae.utils.create_example import create_example
from ...data import triangle_graph_blocking, st_varent_network

# Adds the folder with the data to the path and collects the paths to the files
lib_path = os.path.abspath(os.path.join("..", "../tests"))
sys.path.append(lib_path)

origin = 5
dest = 13


# These test networks are small enough that its expected that A* and Dijkstra's algorithm return the same paths
class TestPathResults(TestCase):
    def setUp(self) -> None:
        self.project = create_example(join(gettempdir(), "test_set_pce_" + uuid4().hex))
        self.project.network.build_graphs()
        self.g = self.project.network.graphs["c"]  # type: Graph
        self.g.set_graph("free_flow_time")
        self.g.set_blocked_centroid_flows(False)

        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        self.r = PathResults()
        self.r.prepare(self.g)

    def tearDown(self) -> None:
        self.project.close()
        self.matrix.close()
        del self.r

    def test_reset(self):
        self.r.compute_path(dest, origin, early_exit=True, a_star=True, heuristic="haversine")
        self.r.reset()

        self.assertEqual(self.r.path, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.path_nodes, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.path_link_directions, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.milepost, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.predecessors.max(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.predecessors.min(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.connectors.max(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.connectors.min(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.early_exit, False, "Fail to reset the Path computation object")
        self.assertEqual(self.r._early_exit, False, "Fail to reset the Path computation object")
        self.assertEqual(self.r.a_star, False, "Fail to reset the Path computation object")
        self.assertEqual(self.r._a_star, False, "Fail to reset the Path computation object")
        self.assertEqual(self.r._heuristic, "equirectangular", "Fail to reset the Path computation object")
        if self.r.skims is not None:
            self.assertEqual(self.r.skims.max(), np.inf, "Fail to reset the Path computation object")
            self.assertEqual(self.r.skims.min(), np.inf, "Fail to reset the Path computation object")

        new_r = PathResults()
        with self.assertRaises(ValueError):
            new_r.reset()

    def test_heuristics(self):
        self.assertEqual(self.r.get_heuristics(), ["haversine", "equirectangular"])

        self.r.set_heuristic("haversine")
        self.assertEqual(self.r._heuristic, "haversine")

        self.r.set_heuristic("equirectangular")
        self.assertEqual(self.r._heuristic, "equirectangular")

    def test_compute_paths(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.early_exit = early_exit
                self.r.a_star = a_star
                path_computation(5, 2, self.g, self.r)

                self.assertEqual(list(self.r.path), [12, 14], "Path computation failed. Wrong sequence of links")
                self.assertEqual(
                    list(self.r.path_link_directions), [1, 1], "Path computation failed. Wrong link directions"
                )
                self.assertEqual(
                    list(self.r.path_nodes), [5, 6, 2], "Path computation failed. Wrong sequence of path nodes"
                )
                self.assertEqual(list(self.r.milepost), [0, 4, 9], "Path computation failed. Wrong milepost results")

    def test_compute_with_skimming(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                r = PathResults()
                self.g.set_skimming("free_flow_time")
                r.prepare(self.g)
                r.compute_path(origin, dest, early_exit=early_exit, a_star=a_star)
                self.assertEqual(r.milepost[-1], r.skims[dest], "Skims computed wrong when computing path")

    def test_update_trace(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.compute_path(origin, 2, early_exit=early_exit, a_star=a_star)

                self.r.update_trace(10)
                self.assertEqual(list(self.r.path), [13, 25], "Path update failed. Wrong sequence of links")
                self.assertEqual(list(self.r.path_link_directions), [1, 1], "Path update failed. Wrong link directions")
                self.assertEqual(
                    list(self.r.path_nodes), [5, 9, 10], "Path update failed. Wrong sequence of path nodes"
                )
                self.assertEqual(list(self.r.milepost), [0, 5, 8], "Path update failed. Wrong milepost results")


class TestBlockingTrianglePathResults(TestCase):
    """
    Triangle blocking network:

                    3 <---> 6
                    ^
            |-------|------|
            |              |
            |              v
    4 <---> 1 <----------- 2 <---> 5

    Graph (link_id, cost):
    1 --> 3: (1, 2)
    3 <-- 2: (2, 4)
    1 <-- 2: (3, 5)
    1 <-> 4: (4, small)
    2 <-> 5: (5, small)
    6 <-> 3: (6, small)

    Geographically all nodes lay on a line.
    """

    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(triangle_graph_blocking, self.proj_dir)
        self.project = Project()
        self.project.open(self.proj_dir)
        self.project.network.build_graphs(modes=["c"])
        self.g = self.project.network.graphs["c"]  # type: Graph
        self.g.set_graph("free_flow_time")
        self.g.set_blocked_centroid_flows(True)

        self.r = PathResults()
        self.r.prepare(self.g)

    def tearDown(self) -> None:
        self.project.close()
        del self.r

    def test_compute_paths(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
                self.assertEqual(list(self.r.path), [1, 2])

                self.r.compute_path(2, 1, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [2, 1])
                self.assertEqual(list(self.r.path), [3])

                self.r.compute_path(3, 1, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [3, 2, 1])
                self.assertEqual(list(self.r.path), [2, 3])

                self.r.compute_path(3, 2, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [3, 2])
                self.assertEqual(list(self.r.path), [2])

                self.r.compute_path(1, 3, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [1, 3])
                self.assertEqual(list(self.r.path), [1])

                self.r.compute_path(2, 3, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [2, 1, 3])
                self.assertEqual(list(self.r.path), [3, 1])

    def test_compute_blocking_paths(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.compute_path(4, 5, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 2, 5])
                self.assertEqual(list(self.r.path), [4, 1, 2, 5])

                self.r.compute_path(5, 4, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [5, 2, 1, 4])
                self.assertEqual(list(self.r.path), [5, 3, 4])

                self.r.compute_path(6, 4, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [6, 3, 2, 1, 4])
                self.assertEqual(list(self.r.path), [6, 2, 3, 4])

                self.r.compute_path(6, 5, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [6, 3, 2, 5])
                self.assertEqual(list(self.r.path), [6, 2, 5])

                self.r.compute_path(4, 6, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 6])
                self.assertEqual(list(self.r.path), [4, 1, 6])

                self.r.compute_path(5, 6, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [5, 2, 1, 3, 6])
                self.assertEqual(list(self.r.path), [5, 3, 1, 6])

    def test_update_trace(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
                self.assertEqual(list(self.r.path), [1, 2])

                self.r.update_trace(3)
                self.assertEqual(list(self.r.path_nodes), [1, 3])
                self.assertEqual(list(self.r.path), [1])

    def test_update_blocking_trace(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.r.compute_path(4, 5, early_exit=early_exit, a_star=a_star)
                self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 2, 5])
                self.assertEqual(list(self.r.path), [4, 1, 2, 5])

                self.r.update_trace(6)
                self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 6])
                self.assertEqual(list(self.r.path), [4, 1, 6])

    def test_update_trace_early_exit(self):
        self.r.compute_path(1, 6, early_exit=True)
        self.assertEqual(list(self.r.path_nodes), [1, 3, 6])
        self.assertEqual(list(self.r.path), [1, 6])

        # Check the state of the partial shortest path tree. While Dijkstra's does explore and find the
        # optimal cost for node 2, it can't prove it's optimal without exploring all of 6's adjacent nodes,
        # which it doesn't do as it exists early upon finding 6
        self.assertEqual(
            [self.r.graph.all_nodes[x] if x != -1 else -1 for x in self.r.predecessors],
            [1, -1, 3, -1, -1, 1, -1],  # Node ids: 4, 5, 6, 1, 2, 3, sentinel
        )

        # Updating to 2 should cause the recomputation of the tree
        self.r.early_exit = True
        self.r.update_trace(2)
        self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
        self.assertEqual(list(self.r.path), [1, 2])

        # The new partial tree state should only have 1 and 5 as -1
        self.assertEqual(
            [self.r.graph.all_nodes[x] if x != -1 else -1 for x in self.r.predecessors],
            [1, -1, 3, -1, 3, 1, -1],  # Node ids: 4, 5, 6, 1, 2, 3, sentinel
        )

    def test_update_trace_full(self):
        self.r.compute_path(1, 6, early_exit=True)

        # Updating to 2 should cause the recomputation of the tree
        self.r.early_exit = False
        self.r.update_trace(2)
        self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
        self.assertEqual(list(self.r.path), [1, 2])

        # The new tree state should be the full tree
        self.assertEqual(
            [self.r.graph.all_nodes[x] if x != -1 else -1 for x in self.r.predecessors],
            [1, 2, 3, -1, 3, 1, -1],  # Node ids: 4, 5, 6, 1, 2, 3, sentinel
        )


class TestCentroidsLast(TestCase):
    def test_compute_paths_centroid_last_node_id(self):
        # Issue 307 consisted of not being able to compute paths between two
        # centroids if we were skimming and one of them had the highest node_id
        # in the entire network
        zipfile.ZipFile(st_varent_network).extractall(gettempdir())
        self.st_varent = join(gettempdir(), "St_Varent")
        self.project = Project()
        self.project.open(self.st_varent)
        self.project.network.build_graphs()
        self.g = self.project.network.graphs["c"]  # type: Graph
        self.g.set_graph("distance")
        self.g.set_skimming("distance")

        self.r = PathResults()
        self.r.prepare(self.g)

        self.r.compute_path(387, 1067)
        self.project.close()
