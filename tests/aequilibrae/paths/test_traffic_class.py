from unittest import TestCase
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
from aequilibrae.paths import TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.utils.create_example import create_example


class TestTrafficClass(TestCase):
    def test_set_pce(self):
        project = create_example(join(gettempdir(), "test_set_pce_" + uuid4().hex))
        project.network.build_graphs()
        car_graph = project.network.graphs["c"]  # type: Graph
        car_graph.set_graph("distance")
        car_graph.set_blocked_centroid_flows(False)

        matrix = project.matrices.get_matrix("demand_omx")
        matrix.computational_view()

        tc = TrafficClass(graph=car_graph, matrix=matrix)

        self.assertIsInstance(tc.results, AssignmentResults, "Results have the wrong type")
        self.assertIsInstance(tc._aon_results, AssignmentResults, "Results have the wrong type")

        with self.assertRaises(ValueError):
            tc.set_pce("not a number")
        tc.set_pce(1)
        tc.set_pce(3.9)
        project.close()
