import os
import sys
from unittest import TestCase

from aequilibrae.paths import path_computation, Graph
from aequilibrae.paths.results import PathResults
from ...data import test_graph
import numpy as np

# Adds the folder with the data to the path and collects the paths to the files
lib_path = os.path.abspath(os.path.join("..", "../tests"))
sys.path.append(lib_path)

origin = 5
dest = 27


class TestPathResults(TestCase):
    def setUp(self) -> None:
        # graph
        self.g = Graph()
        self.g.load_from_disk(test_graph)
        self.g.set_graph(cost_field="distance")

        self.r = PathResults()
        try:
            self.r.prepare(self.g)
        except Exception as err:
            self.fail("Path result preparation failed - {}".format(err.__str__()))

    def test_reset(self):
        self.r.compute_path(dest, origin)
        self.r.reset()

        self.assertEqual(self.r.path, None, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.path_nodes, None, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.path_link_directions, None, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.milepost, None, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.predecessors.max(), -1, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.predecessors.min(), -1, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.connectors.max(), -1, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.connectors.min(), -1, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.skims.max(), np.inf, 'Fail to reset the Path computation object')
        self.assertEqual(self.r.skims.min(), np.inf, 'Fail to reset the Path computation object')

        new_r = PathResults()
        with self.assertRaises(ValueError):
            new_r.reset()

    def test_compute_paths(self):

        path_computation(origin, dest, self.g, self.r)

        if list(self.r.path) != [53, 52, 13]:
            self.fail("Path computation failed. Wrong sequence of links")

        if list(self.r.path_nodes) != [5, 168, 166, 27]:
            self.fail("Path computation failed. Wrong sequence of path nodes")

        if list(self.r.milepost) != [0, 341, 1398, 2162]:
            self.fail("Path computation failed. Wrong milepost results")

        self.r.compute_path(origin, dest)

        if list(self.r.path) != [53, 52, 13]:
            self.fail("Path computation failed. Wrong sequence of links")

        if list(self.r.path_nodes) != [5, 168, 166, 27]:
            self.fail("Path computation failed. Wrong sequence of path nodes")

        if list(self.r.milepost) != [0, 341, 1398, 2162]:
            self.fail("Path computation failed. Wrong milepost results")

        if list(self.r.path_link_directions) != [-1, -1, -1]:
            self.fail("Path computation failed. Wrong link directions")

        self.r.compute_path(dest, origin)
        if list(self.r.path_link_directions) != [1, 1, 1]:
            self.fail("Path computation failed. Wrong link directions")

    def test_update_trace(self):
        self.r.compute_path(origin, dest - 1)

        self.r.update_trace(dest)

        if list(self.r.path) != [53, 52, 13]:
            self.fail("Path computation failed. Wrong sequence of links")

        if list(self.r.path_nodes) != [5, 168, 166, 27]:
            self.fail("Path computation failed. Wrong sequence of path nodes")

        if list(self.r.milepost) != [0, 341, 1398, 2162]:
            self.fail("Path computation failed. Wrong milepost results")

        if list(self.r.path_link_directions) != [-1, -1, -1]:
            self.fail("Path computation failed. Wrong link directions")
