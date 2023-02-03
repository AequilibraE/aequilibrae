import os
from tempfile import gettempdir
from unittest import TestCase
import uuid
from aequilibrae.utils.create_example import create_example
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.all_or_nothing import allOrNothing
from ...data import test_graph


# TODO: Add checks for results for this test (Assignment AoN)
class TestAllOrNothing(TestCase):
    def setUp(self) -> None:
        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_" + uuid.uuid4().hex)
        self.project = create_example(proj_path)
        self.project.network.build_graphs()
        self.g = self.project.network.graphs["c"]  # type: Graph
        self.g.set_graph("distance")
        self.g.set_skimming("distance")
        self.g.set_blocked_centroid_flows(False)

        self.matrix = self.project.matrices.get_matrix("demand_aem")
        self.matrix.computational_view()
        self.matrix2 = self.project.matrices.get_matrix("demand_omx")
        self.matrix2.computational_view()
        self.matrix2.matrix_view *= 2

    def tearDown(self) -> None:
        self.matrix.close()
        self.matrix2.close()
        self.project.close()

    def test_skimming_on_assignment(self):
        res = AssignmentResults()

        res.prepare(self.g, self.matrix)

        self.g.set_skimming([])
        self.g.set_blocked_centroid_flows(True)
        assig = allOrNothing(self.matrix, self.g, res)
        assig.execute()

        if res.skims.distance.sum() > 0:
            self.fail("skimming for nothing during assignment returned something different than zero")

        res.prepare(self.g, self.matrix)

        assig = allOrNothing(self.matrix, self.g, res)
        assig.execute()

    def test_execute(self):
        # Loads and prepares the graph
        res1 = AssignmentResults()
        res1.prepare(self.g, self.matrix)
        assig1 = allOrNothing(self.matrix, self.g, res1)
        assig1.execute()

        res2 = AssignmentResults()
        res2.prepare(self.g, self.matrix2)
        assig2 = allOrNothing(self.matrix2, self.g, res2)
        assig2.execute()

        load1 = res1.get_load_results()
        load2 = res2.get_load_results()

        self.assertEqual(list(load1.matrix_tot * 2), list(load2.matrix_tot), "Something wrong with the AoN")
