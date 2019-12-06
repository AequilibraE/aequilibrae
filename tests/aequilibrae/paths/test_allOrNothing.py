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

        # Creates the matrix for assignment
        args = {
            "file_name": os.path.join(gettempdir(), "my_matrix.aem"),
            "zones": g.num_zones,
            "matrix_names": ["cars", "trucks"],
            "index_names": ["my indices"],
        }

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = g.centroids[:]
        matrix.cars.fill(1.1)
        matrix.trucks.fill(2.2)

        # Exports matrix to OMX in order to have two matrices to work with
        matrix.export(os.path.join(gettempdir(), "my_matrix.omx"))
        matrix.close()

        car_loads = []
        two_class_loads = []
        for extension in ["omx", "aem"]:
            matrix = AequilibraeMatrix()
            matrix.load(os.path.join(gettempdir(), "my_matrix." + extension))

            matrix.computational_view(["cars"])

            # Performs assignment
            res = AssignmentResults()
            res.prepare(g, matrix)

            assig = allOrNothing(matrix, g, res)
            assig.execute()
            car_loads.append(res.link_loads)
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_{}.aed".format(extension)))
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_{}.csv".format(extension)))

            matrix.computational_view()
            # Performs assignment
            res = AssignmentResults()
            res.prepare(g, matrix)

            assig = allOrNothing(matrix, g, res)
            assig.execute()
            two_class_loads.append(res.link_loads)
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes_{}.aed".format(extension)))
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes_{}.csv".format(extension)))

        load_diff = two_class_loads[0] - two_class_loads[1]
        if load_diff.max() > 0.0000000001 or load_diff.max() < -0.0000000001:
            self.fail("Loads for two classes differ for OMX and AEM matrix types")

        load_diff = car_loads[0] - car_loads[1]
        if load_diff.max() > 0.0000000001 or load_diff.max() < -0.0000000001:
            self.fail("Loads for a single class differ for OMX and AEM matrix types")
