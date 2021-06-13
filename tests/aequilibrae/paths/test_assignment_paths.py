from unittest import TestCase
import os
import pandas as pd
import numpy as np
import uuid
from tempfile import gettempdir
from aequilibrae import TrafficAssignment, TrafficClass, Graph
from aequilibrae.paths.assignment_paths import AssignmentPaths, AssignmentResultsTable, TrafficClassIdentifier
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
        self.result_name = "ass_path_test"
        self.assignment.save_results(self.result_name)
        # pathlib.Path(self.project.project_base_path)

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_assignment_results_table(self):
        ass_res_table = AssignmentResultsTable(self.result_name)

        self.assertEqual(ass_res_table.table_name, self.result_name)
        self.assertEqual(ass_res_table.procedure, "traffic assignment")
        self.assertIsInstance(ass_res_table.assignment_results, pd.DataFrame)

        traffic_class_id_and_name = ass_res_table.get_traffic_class_names_and_id()
        self.assertEqual(len(traffic_class_id_and_name), 1)
        self.assertIsInstance(traffic_class_id_and_name[0], TrafficClassIdentifier)
        self.assertEqual(traffic_class_id_and_name[0].mode, self.network_mode)
        self.assertEqual(traffic_class_id_and_name[0].__id__, self.traffic_class_name)

        self.assertEqual(ass_res_table.get_number_of_iterations(), 1)
        self.assertEqual(ass_res_table.get_assignment_method(), "msa")
        car_class_info = ass_res_table.procedure_report["setup"]["Classes"][self.traffic_class_name]
        car_class_keys = list(car_class_info.keys())
        self.assertIn("not_assigned", car_class_keys)
        self.assertIn("matrix_totals", car_class_keys)
        self.assertIn("intrazonals", car_class_keys)
        self.assertEqual(car_class_info["network mode"], self.network_mode)
        self.assertEqual(car_class_info["Value-of-time"], 1.0)
        self.assertEqual(car_class_info["PCE"], 1.0)
        self.assertEqual(car_class_info["save_path_files"], True)
        self.assertEqual(car_class_info["path_file_feather_format"], True)

        convergence_info = ass_res_table.procedure_report["convergence"]
        self.assertEqual(ass_res_table.get_alphas(), [1.0])
        self.assertEqual(convergence_info["iteration"], [1])
        self.assertEqual(convergence_info["rgap"], [np.inf])
        self.assertEqual(convergence_info["warnings"], [""])

    def test_assignment_path_reader(self):
        paths = AssignmentPaths(self.result_name)
        self.assertEqual(paths.proj_dir, self.project.project_base_path)
        self.assertIsInstance(paths.assignment_results, AssignmentResultsTable)
        self.assertEqual(len(paths.classes), 1)
        self.assertIsInstance(paths.classes[0], TrafficClassIdentifier)
        self.assertEqual(len(paths.compressed_graph_correspondences), 1)
        self.assertIsInstance(paths.compressed_graph_correspondences[self.traffic_class_name], pd.DataFrame)

        o, d, iteration = 0, 1, 1
        path_0_1 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_0_1), [0])

        # following is taken from path computation tests, we use compressed_id, which for SF is link_id - 1
        # and node index, which here is node_nr - 1.
        o, d = 4, 1
        path_4_1 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_4_1), [11, 13])

        o, d = 4, 9
        path_4_9 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_4_9), [12, 24])

        o, d = 9, 4
        path_9_4 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_9_4), [25, 22])

        # Let's try some longer paths, and try directionality: 17->18 does not exist, but 18->17 does
        o, d = 5, 17
        path_5_17 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_5_17), [15, 19, 17])
        o, d = 5, 18
        path_5_18 = paths.get_path_for_destination(o, d, iteration, self.traffic_class_name)
        self.assertEqual(list(path_5_18), [15, 21, 48, 52])

        p1, p2 = paths.read_path_file(o, iteration, self.traffic_class_name)
        self.assertIsInstance(p1, pd.DataFrame)
        self.assertIsInstance(p2, pd.DataFrame)
        p_5_18 = paths.get_path_for_destination_from_files(p1, p2, d)
        self.assertEqual(list(path_5_18), list(p_5_18))
