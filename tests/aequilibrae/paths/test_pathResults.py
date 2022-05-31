import os
import sys
import uuid
import zipfile
from os.path import join
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4

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
        self.r.compute_path(dest, origin)
        self.r.reset()

        self.assertEqual(self.r.path, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.path_nodes, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.path_link_directions, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.milepost, None, "Fail to reset the Path computation object")
        self.assertEqual(self.r.predecessors.max(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.predecessors.min(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.connectors.max(), -1, "Fail to reset the Path computation object")
        self.assertEqual(self.r.connectors.min(), -1, "Fail to reset the Path computation object")
        if self.r.skims is not None:
            self.assertEqual(self.r.skims.max(), np.inf, "Fail to reset the Path computation object")
            self.assertEqual(self.r.skims.min(), np.inf, "Fail to reset the Path computation object")

        new_r = PathResults()
        with self.assertRaises(ValueError):
            new_r.reset()

    def test_compute_paths(self):
        path_computation(5, 2, self.g, self.r)

        self.assertEqual(list(self.r.path), [12, 14], "Path computation failed. Wrong sequence of links")
        self.assertEqual(list(self.r.path_link_directions), [1, 1], "Path computation failed. Wrong link directions")
        self.assertEqual(list(self.r.path_nodes), [5, 6, 2], "Path computation failed. Wrong sequence of path nodes")
        self.assertEqual(list(self.r.milepost), [0, 4, 9], "Path computation failed. Wrong milepost results")

    def test_compute_with_skimming(self):
        r = PathResults()
        self.g.set_skimming("free_flow_time")
        r.prepare(self.g)
        r.compute_path(origin, dest)
        self.assertEqual(r.milepost[-1], r.skims[dest], "Skims computed wrong when computing path")

    def test_update_trace(self):
        self.r.compute_path(origin, 2)

        self.r.update_trace(10)
        self.assertEqual(list(self.r.path), [13, 25], "Path update failed. Wrong sequence of links")
        self.assertEqual(list(self.r.path_link_directions), [1, 1], "Path update failed. Wrong link directions")
        self.assertEqual(list(self.r.path_nodes), [5, 9, 10], "Path update failed. Wrong sequence of path nodes")
        self.assertEqual(list(self.r.milepost), [0, 5, 8], "Path update failed. Wrong milepost results")


class TestBlockingTrianglePathResults(TestCase):
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
        self.r.compute_path(1, 2)
        self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
        self.assertEqual(list(self.r.path), [1, 2])

        self.r.compute_path(2, 1)
        self.assertEqual(list(self.r.path_nodes), [2, 1])
        self.assertEqual(list(self.r.path), [3])

        self.r.compute_path(3, 1)
        self.assertEqual(list(self.r.path_nodes), [3, 2, 1])
        self.assertEqual(list(self.r.path), [2, 3])

        self.r.compute_path(3, 2)
        self.assertEqual(list(self.r.path_nodes), [3, 2])
        self.assertEqual(list(self.r.path), [2])

        self.r.compute_path(1, 3)
        self.assertEqual(list(self.r.path_nodes), [1, 3])
        self.assertEqual(list(self.r.path), [1])

        self.r.compute_path(2, 3)
        self.assertEqual(list(self.r.path_nodes), [2, 1, 3])
        self.assertEqual(list(self.r.path), [3, 1])

    def test_compute_blocking_paths(self):
        self.r.compute_path(4, 5)
        self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 2, 5])
        self.assertEqual(list(self.r.path), [4, 1, 2, 5])

        self.r.compute_path(5, 4)
        self.assertEqual(list(self.r.path_nodes), [5, 2, 1, 4])
        self.assertEqual(list(self.r.path), [5, 3, 4])

        self.r.compute_path(6, 4)
        self.assertEqual(list(self.r.path_nodes), [6, 3, 2, 1, 4])
        self.assertEqual(list(self.r.path), [6, 2, 3, 4])

        self.r.compute_path(6, 5)
        self.assertEqual(list(self.r.path_nodes), [6, 3, 2, 5])
        self.assertEqual(list(self.r.path), [6, 2, 5])

        self.r.compute_path(4, 6)
        self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 6])
        self.assertEqual(list(self.r.path), [4, 1, 6])

        self.r.compute_path(5, 6)
        self.assertEqual(list(self.r.path_nodes), [5, 2, 1, 3, 6])
        self.assertEqual(list(self.r.path), [5, 3, 1, 6])

    def test_update_trace(self):
        self.r.compute_path(1, 2)
        self.assertEqual(list(self.r.path_nodes), [1, 3, 2])
        self.assertEqual(list(self.r.path), [1, 2])

        self.r.update_trace(3)
        self.assertEqual(list(self.r.path_nodes), [1, 3])
        self.assertEqual(list(self.r.path), [1])

    def test_update_blocking_trace(self):
        self.r.compute_path(4, 5)
        self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 2, 5])
        self.assertEqual(list(self.r.path), [4, 1, 2, 5])

        self.r.update_trace(6)
        self.assertEqual(list(self.r.path_nodes), [4, 1, 3, 6])
        self.assertEqual(list(self.r.path), [4, 1, 6])


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
