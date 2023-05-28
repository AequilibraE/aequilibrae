import os
import sqlite3
import uuid
import zipfile
from os.path import dirname
from os.path import join
from tempfile import gettempdir
from unittest import TestCase

import numpy as np
import pandas as pd

from aequilibrae import Graph
from aequilibrae import Project
from aequilibrae import TrafficAssignment
from aequilibrae import TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.results.turn_volumes_results import TurnVolumesResults
from aequilibrae.utils.spatialite_utils import ensure_spatialite_binaries
from ...data import siouxfalls_project

TURNS_DF = pd.DataFrame([[1, 2, 6]], columns=["a", "b", "c"])


class TestTurnVolumes(TestCase):
    def setUp(self) -> None:
        ensure_spatialite_binaries()

        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_path_files" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)

        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix_1 = self._create_assign_matrix()
        self.matrix_1.computational_view(["mat1"])

        self.matrix_2 = self._create_assign_matrix()
        self.matrix_2.computational_view(["mat2"])

        self.assignment = TrafficAssignment()
        self.assignclass_1 = TrafficClass("car", self.car_graph, self.matrix_1)
        self.assignclass_2 = TrafficClass("truck", self.car_graph, self.matrix_2)
        self.assignment.set_classes([self.assignclass_1, self.assignclass_2])
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

    def tearDown(self) -> None:
        self.matrix_1.close()
        self.matrix_2.close()
        self.project.close()

    def _create_assign_matrix(self):
        # The matrices are designed to create congestion.
        # Trips to and from centroid 2 are set to 0.
        # Turn volumes for turn 1,2,6 should be the same as the volumes on link id 1
        zones = 24
        mat_file = join(gettempdir(), f"Aequilibrae_matrix_{uuid.uuid4()}.aem")
        args = {
            "file_name": mat_file,
            "zones": zones,
            "matrix_names": ["mat1", "mat2"],
            "index_names": ["my_indices"],
            "memory_only": False,
        }
        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)
        trips = np.ones((zones, zones)) * 500
        trips[[1, 5], :] = 0
        trips[:, [1, 5]] = 0

        matrix.index[:] = np.arange(matrix.zones) + 1
        matrix.matrices[:, :, 0] = trips
        matrix.matrices[:, :, 1] = matrix.mat1 * 2
        matrix.setName("test_turn_volumes")
        matrix.setDescription("test turn movements")
        matrix.matrix_hash = {zone: idx for idx, zone in enumerate(np.arange(matrix.zones) + 1)}
        return matrix

    def test_all_or_nothing_from_asgn(self):
        """
        Tests whether the turn volumes works as intended from an all or nothing assignment
        Uses 2 classes with trips between 2 overlapping OD pairs
        """
        self.assignment.set_algorithm("all-or-nothing")
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        results_df = self.assignment.results()

        turning_movements = self.assignment.turning_volumes(TURNS_DF)
        self.assertEqual(turning_movements.at[0, "volume"], results_df.at[1, "mat1_ab"])
        self.assertEqual(turning_movements.at[1, "volume"], results_df.at[1, "mat2_ab"])

    def test_all_or_nothing_from_results(self):
        """
        Tests whether the turn volumes works as intended from an all or nothing existing results
        Uses 2 classes with trips between 2 overlapping OD pairs
        """
        # Results tables in test projects don't contain all required info.
        # Must reassign to create an up-to-date results table.

        self.assignment.set_algorithm("all-or-nothing")
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        self.assignment.save_results("test_turn_movements")
        results_df = self.assignment.results()
        class_to_matrix = {"car": self.matrix_1, "truck": self.matrix_2}
        turning_movements = TurnVolumesResults.calculate_from_result_table(
            project=self.project,
            turns_df=TURNS_DF,
            asgn_result_table_name="test_turn_movements",
            class_to_matrix=class_to_matrix,
        )
        self.assertEqual(turning_movements.at[0, "volume"], results_df.at[1, "mat1_ab"])
        self.assertEqual(turning_movements.at[1, "volume"], results_df.at[1, "mat2_ab"])

    def test_bfw_from_asgn(self):
        """
        Tests whether the turn volumes works as intended from a bfw assignment
        Uses 2 classes with trips between 2 overlapping OD pairs
        """
        self.assignment.set_algorithm("bfw")
        self.assignment.max_iter = 5
        self.assignment.rgap_target = 0.001
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        results_df = self.assignment.results()
        turning_movements = self.assignment.turning_volumes(TURNS_DF)
        self.assertEqual(turning_movements.at[0, "volume"], results_df.at[1, "mat1_ab"])
        self.assertEqual(turning_movements.at[1, "volume"], results_df.at[1, "mat2_ab"])

    def test_bfw_from_results(self):
        """
        Tests whether the turn volumes works as intended from a bfw existing results
        Uses 2 classes with trips between 2 overlapping OD pairs
        """
        # Results tables in test projects don't contain all required info.
        # Must reassign to create an up-to-date results table.

        self.assignment.set_algorithm("bfw")
        self.assignment.max_iter = 5
        self.assignment.rgap_target = 0.001
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        self.assignment.save_results("test_turn_movements")
        results_df = self.assignment.results()

        class_to_matrix = {"car": self.matrix_1, "truck": self.matrix_2}
        turning_movements = TurnVolumesResults.calculate_from_result_table(
            project=self.project,
            turns_df=TURNS_DF,
            asgn_result_table_name="test_turn_movements",
            class_to_matrix=class_to_matrix,
        )

        self.assertEqual(turning_movements.at[0, "volume"], results_df.at[1, "mat1_ab"])
        self.assertEqual(turning_movements.at[1, "volume"], results_df.at[1, "mat2_ab"])

    def test_save_turn_volumes(self):
        """
        Tests whether the turn volumes are saved to the results database
        """
        self.assignment.set_algorithm("all-or-nothing")
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        turn_volumes_table_name = "test_turn_movements"
        turning_movements = self.assignment.save_turning_volumes(turn_volumes_table_name, TURNS_DF)
        conn = sqlite3.connect(os.path.join(self.project.project_base_path, "results_database.sqlite"))
        df = pd.read_sql_query(f"select * from {turn_volumes_table_name}", conn).drop(columns="index")
        conn.close()
        self.assertIsNone(pd.testing.assert_frame_equal(turning_movements, df))