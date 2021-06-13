from unittest import TestCase
import os
import uuid
from tempfile import gettempdir
from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths.select_link import SelectLink
from aequilibrae.utils.create_example import create_example

# TODO (Jan 13/6/21): Do we want to add a result table to the SiouxFalls project in tests/data? Or maybe in the
#  reference project? This here depends on an assignment run.


class TestAssignmentPaths(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_assignment_paths_" + uuid.uuid4().hex)
        self.project = create_example(proj_path)
        self.project.network.build_graphs()
        self.network_mode = "c"
        self.car_graph = self.project.network.graphs[self.network_mode]
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        self.assignment = TrafficAssignment()
        self.traffic_class_name = "car"
        self.assigclass = TrafficClass(self.traffic_class_name, self.car_graph, self.matrix)
        self.assignment.add_class(self.assigclass)
        self.assignment.set_save_path_files(True)
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")
        self.assignment.max_iter = 1
        self.assignment.set_algorithm("msa")
        self.assignment.execute()
        self.result_name = "sl_test"
        self.assignment.save_results(self.result_name)

        self.golden_non_zero_vals = [
            (0, 1),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (2, 1),
            (11, 1),
            (12, 1),
        ]
        self.golden_sum = 3800.0

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    # TODO Jan (13/6/21): need tests for various methods called during instantiation. Maybe implement default
    #  constructor just for tests and then go from there. Alternatively, just use the test project.

    def test_run_select_link_analysis(self):
        demand_mat = {"car": self.matrix}
        sl = SelectLink(self.result_name, demand_mat)
        test_link = [0]
        sl_res = sl.run_select_link_analysis(test_link)
        res = sl_res[self.traffic_class_name][0]
        self.assertEqual(len(sl_res), 1)
        self.assertEqual(len(sl_res[self.traffic_class_name]), 1)

        self.assertEqual(res.sum(), self.golden_sum)
        self.assertEqual(self.matrix.matrix_view[res.nonzero()].sum(), res.sum())

        non_zero_matrix_vals = list(zip(res.nonzero()[0], res.nonzero()[1]))
        self.assertEqual(non_zero_matrix_vals, self.golden_non_zero_vals)
