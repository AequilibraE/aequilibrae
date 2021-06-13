from unittest import TestCase
import os
import pathlib
import sqlite3
import pandas as pd
import numpy as np
import uuid
import string
import random
from random import choice
from tempfile import gettempdir
from aequilibrae import TrafficAssignment, TrafficClass, Graph
from aequilibrae.paths.assignment_paths import AssignmentPaths, AssignmentResultsTable, TrafficClassIdentifier
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project

# TODO (Jan 13/6/21): Do we want to add a result table to the SiouxFalls project in tests/data? Or maybe in the
#  reference project? This here depends on an assignment run, not ideal.


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

    #
    #
    # def test_read_assignment_results(self):
    #
    # reference_path_file_dir = pathlib.Path(siouxfalls_project) / "path_files"
    #
    # ref_node_correspondence = pd.read_feather(reference_path_file_dir / f"nodes_to_indeces_{class_id}.feather")
    # node_correspondence = pd.read_feather(path_file_dir / f"nodes_to_indeces_{class_id}.feather")
    # self.assertTrue(node_correspondence.equals(ref_node_correspondence))
    #
    # ref_correspondence = pd.read_feather(reference_path_file_dir / f"correspondence_{class_id}.feather")
    # correspondence = pd.read_feather(path_file_dir / f"correspondence_{class_id}.feather")
    # self.assertTrue(correspondence.equals(ref_correspondence))
    #
    # path_class_id = f"path_{class_id}"
    # for i in range(1, self.assignment.max_iter + 1):
    #     class_dir = path_file_dir / f"iter{i}" / path_class_id
    #     ref_class_dir = reference_path_file_dir / f"iter{i}" / path_class_id
    #     for o in self.assigclass.matrix.index:
    #         o_ind = self.assigclass.graph.compact_nodes_to_indices[o]
    #         this_o_path_file = pd.read_feather(class_dir / f"o{o_ind}.feather")
    #         ref_this_o_path_file = pd.read_feather(ref_class_dir / f"o{o_ind}.feather")
    #         is_eq = this_o_path_file == ref_this_o_path_file
    #         self.assertTrue(is_eq.all().all())
    #
    #         this_o_index_file = pd.read_feather(class_dir / f"o{o_ind}_indexdata.feather")
    #         ref_this_o_index_file = pd.read_feather(ref_class_dir / f"o{o_ind}_indexdata.feather")
    #         is_eq = this_o_index_file == ref_this_o_index_file
    #         self.assertTrue(is_eq.all().all())
    #
