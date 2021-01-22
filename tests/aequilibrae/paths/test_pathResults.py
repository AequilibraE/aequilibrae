import os
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
import sys
from unittest import TestCase

from aequilibrae.paths import path_computation, Graph
from aequilibrae.paths.results import PathResults
from aequilibrae.paths import binary_version
from aequilibrae.utils.create_example import create_example
from ...data import test_graph
import numpy as np

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
