import os
from tempfile import gettempdir
from unittest import TestCase

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import allOrNothing
from aequilibrae.paths import Graph
from aequilibrae.paths import AssignmentResults
from ...data import test_graph


# TODO: Add checks for results for this test (Assignment AoN)
class TestAllOrNothing(TestCase):
    def test_execute(self):
        # Loads and prepares the graph
        g = Graph()
        g.load_from_disk(test_graph)
        g.set_graph(cost_field="distance", skim_fields=None)
        # None implies that only the cost field will be skimmed

        # Prepares the matrix for assignment
        args = {
            "file_name": os.path.join(gettempdir(), "my_matrix.aem"),
            "zones": g.num_zones,
            "matrix_names": ["cars", "trucks"],
            "index_names": ["my indices"],
        }

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = g.centroids[:]
        matrix.cars.fill(1)
        matrix.trucks.fill(2)
        matrix.computational_view(["cars"])

        # Performs assignment
        res = AssignmentResults()
        res.prepare(g, matrix)

        assig = allOrNothing(matrix, g, res)
        assig.execute()

        res.save_to_disk(os.path.join(gettempdir(), "link_loads.aed"))
        res.save_to_disk(os.path.join(gettempdir(), "link_loads.csv"))

        matrix.computational_view()
        # Performs assignment
        res = AssignmentResults()
        res.prepare(g, matrix)

        assig = allOrNothing(matrix, g, res)
        assig.execute()
        res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes.aed"))
        res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes.csv"))
