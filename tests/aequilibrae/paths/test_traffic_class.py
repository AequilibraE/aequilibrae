from unittest import TestCase
from aequilibrae.paths import TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults

from tempfile import gettempdir
import os
from ...data import test_graph


class TestTrafficClass(TestCase):
    def test_set_pce(self):
        mat_name = AequilibraeMatrix().random_name()
        g = Graph()
        g.load_from_disk(test_graph)
        g.set_graph(cost_field="distance")

        # Creates the matrix for assignment
        args = {
            "file_name": os.path.join(gettempdir(), mat_name),
            "zones": g.num_zones,
            "matrix_names": ["cars", "trucks"],
            "index_names": ["my indices"],
        }

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = g.centroids[:]
        matrix.cars.fill(1.1)
        matrix.trucks.fill(2.2)
        matrix.computational_view()

        tc = TrafficClass(graph=g, matrix=matrix)

        self.assertIsInstance(tc.results, AssignmentResults, 'Results have the wrong type')
        self.assertIsInstance(tc._aon_results, AssignmentResults, 'Results have the wrong type')

        with self.assertRaises(ValueError):
            tc.set_pce('not a number')
        tc.set_pce(1)
        tc.set_pce(3.9)
