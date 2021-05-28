from unittest import TestCase
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
from aequilibrae.paths import TrafficClass
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.utils.create_example import create_example


class TestTrafficClass(TestCase):
    def setUp(self) -> None:
        self.project = create_example(join(gettempdir(), "test_set_pce_" + uuid4().hex))
        self.project.network.build_graphs()
        car_graph = self.project.network.graphs["c"]  # type: Graph
        car_graph.set_graph("distance")
        car_graph.set_blocked_centroid_flows(False)
        matrix = self.project.matrices.get_matrix("demand_omx")
        matrix.computational_view()
        self.tc = TrafficClass(name="car", graph=car_graph, matrix=matrix)

    def tearDown(self) -> None:
        self.project.close()

    def test_result_type(self):
        self.assertIsInstance(self.tc.results, AssignmentResults, "Results have the wrong type")
        self.assertIsInstance(self.tc._aon_results, AssignmentResults, "Results have the wrong type")

    def test_set_pce(self):
        with self.assertRaises(ValueError):
            self.tc.set_pce("not a number")
        self.tc.set_pce(1)
        self.tc.set_pce(3.9)

    def test_set_vot(self):
        self.assertEqual(self.tc.vot, 1.0)
        self.tc.set_vot(4.5)
        self.assertEqual(self.tc.vot, 4.5)

    def test_set_fixed_cost(self):
        self.assertEqual(self.tc.fc_multiplier, 1.0)

        with self.assertRaises(ValueError):
            self.tc.set_fixed_cost("Field_Does_Not_Exist", 2.5)
        self.assertEqual(self.tc.fc_multiplier, 1.0)

        self.tc.set_fixed_cost("distance", 3.0)
        self.assertEqual(self.tc.fc_multiplier, 3.0)
        self.assertEqual(self.tc.fixed_cost_field, "distance")
