import os
from tempfile import gettempdir
from unittest import TestCase
import numpy as np
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.all_or_nothing import allOrNothing
from ...data import test_graph


# TODO: Add checks for results for this test (Assignment AoN)
class TestAllOrNothing(TestCase):
    def setUp(self) -> None:
        self.mat_name = AequilibraeMatrix().random_name()
        self.g = Graph()
        self.g.load_from_disk(test_graph)
        self.g.set_graph(cost_field="distance")

        # Creates the matrix for assignment
        args = {
            "file_name": os.path.join(gettempdir(), self.mat_name),
            "zones": self.g.num_zones,
            "matrix_names": ["cars", "trucks"],
            "index_names": ["my indices"],
        }

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = self.g.centroids[:]
        matrix.cars.fill(1.1)
        matrix.trucks.fill(2.2)

        # Exports matrix to OMX in order to have two matrices to work with
        matrix.export(os.path.join(gettempdir(), "my_matrix.omx"))
        matrix.close()

    def test_skimming_on_assignment(self):
        matrix = AequilibraeMatrix()
        matrix.load(os.path.join(gettempdir(), self.mat_name))
        matrix.computational_view(["cars"])

        res = AssignmentResults()

        res.prepare(self.g, matrix)

        self.g.set_skimming([])
        self.g.set_blocked_centroid_flows(True)
        assig = allOrNothing(matrix, self.g, res)
        assig.execute()

        if res.skims.distance.sum() > 0:
            self.fail("skimming for nothing during assignment returned something different than zero")

        self.g.set_skimming("distance")
        res.prepare(self.g, matrix)

        assig = allOrNothing(matrix, self.g, res)
        assig.execute()
        if res.skims.distance.sum() != 2914644.0:
            self.fail("skimming during assignment returned the wrong value")
        matrix.close()

    def test_execute(self):
        # Loads and prepares the graph

        car_loads = []
        two_class_loads = []

        for extension in ["omx", "aem"]:
            matrix = AequilibraeMatrix()
            if extension == 'omx':
                mat_name = os.path.join(gettempdir(), "my_matrix." + extension)
            else:
                mat_name = self.mat_name
            matrix.load(mat_name)

            matrix.computational_view(["cars"])

            # Performs assignment
            res = AssignmentResults()
            res.prepare(self.g, matrix)

            assig = allOrNothing(matrix, self.g, res)
            assig.execute()
            car_loads.append(res.link_loads)
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_{}.aed".format(extension)))
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_{}.csv".format(extension)))

            matrix.computational_view()
            # Performs assignment
            res = AssignmentResults()
            res.prepare(self.g, matrix)

            assig = allOrNothing(matrix, self.g, res)
            assig.execute()
            two_class_loads.append(res.link_loads)
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes_{}.aed".format(extension)))
            res.save_to_disk(os.path.join(gettempdir(), "link_loads_2_classes_{}.csv".format(extension)))
            matrix.close()

        load_diff = two_class_loads[0] - two_class_loads[1]
        if load_diff.max() > 0.0000000001 or load_diff.max() < -0.0000000001:
            self.fail("Loads for two classes differ for OMX and AEM matrix types")

        load_diff = car_loads[0] - car_loads[1]
        if load_diff.max() > 0.0000000001 or load_diff.max() < -0.0000000001:
            self.fail("Loads for a single class differ for OMX and AEM matrix types")
