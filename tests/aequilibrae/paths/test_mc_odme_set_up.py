import os
import pathlib
import random
import sqlite3
import string
import uuid
from random import choice
from tempfile import gettempdir
from unittest import TestCase

import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, TrafficClass, Graph, ODME
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project


class TestODMEMultiClassSetUp(TestCase):
    """
    Basic tests of ODME algorithm with multiple user classes.

    Currently taken from test_mc_traffic_assignment.py
    """

    def setUp(self) -> None:
        # Download example project
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        # Create graphs
        proj_path = os.path.join(gettempdir(), "test_mc_traffic_assignment_" + uuid.uuid4().hex)
        self.project = create_example(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.truck_graph = self.project.network.graphs["T"]  # type: Graph
        self.moto_graph = self.project.network.graphs["M"]  # type: Graph

        for graph in [self.car_graph, self.truck_graph, self.moto_graph]:
            graph.set_skimming(["free_flow_time"])
            graph.set_graph("free_flow_time")
            graph.set_blocked_centroid_flows(False)

        # Open matrices:
        self.car_matrix = self.project.matrices.get_matrix("demand_mc")
        self.car_matrix.computational_view(["car"])
        self.car_index = self.car_graph.nodes_to_indices

        self.truck_matrix = self.project.matrices.get_matrix("demand_mc")
        self.truck_matrix.computational_view(["trucks"])
        self.truck_index = self.truck_graph.nodes_to_indices

        self.moto_matrix = self.project.matrices.get_matrix("demand_mc")
        self.moto_matrix.computational_view(["motorcycle"])
        self.moto_index = self.moto_graph.nodes_to_indices

        # Create assignment object and assign classes
        self.assignment = TrafficAssignment()
        self.carclass = TrafficClass("car", self.car_graph, self.car_matrix)
        self.carclass.set_pce(1.0)
        self.motoclass = TrafficClass("motorcycle", self.moto_graph, self.moto_matrix)
        self.motoclass.set_pce(0.2)
        self.truckclass = TrafficClass("truck", self.truck_graph, self.truck_matrix)
        self.truckclass.set_pce(2.5)

        self.assignment.set_classes([self.carclass, self.truckclass, self.motoclass])

        # Set assignment parameters
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 5
        self.assignment.set_algorithm("msa")

        # Store parameters needed for ODME/demand matrix manipulation:
        self.count_vol_cols = ["class", "link_id", "direction", "obs_volume"]
        self.car_index = self.car_graph.nodes_to_indices
        self.car_dims = self.car_matrix.matrix_view.shape
        self.truck_index = self.truck_graph.nodes_to_indices
        self.truck_dims = self.truck_matrix.matrix_view.shape
        self.moto_index = self.moto_graph.nodes_to_indices
        self.moto_dims = self.moto_matrix.matrix_view.shape

        self.user_classes = self.assignment.classes
        self.user_class_names = [user_class.__id__ for user_class in self.user_classes]
        self.user_class_dims = [self.car_dims, self.moto_dims, self.truck_dims]
        self.matrices = [user_class.matrix for user_class in self.user_classes]

        # Currently testing algorithm:
        self.algorithm = "spiess"

    def tearDown(self) -> None:
        for mat in [self.car_matrix, self.truck_matrix, self.moto_matrix]:
            mat.close()
        self.project.close()

    def test_playground(self) -> None:
        """
        Used to mess around with various functions and see how things work
        before writing actual tests.

        Should be removed later!
        """
        self.car_matrix.matrix_view = np.zeros(self.user_class_dims[0])
        self.car_matrix.matrix_view[self.car_index[1], self.car_index[2]] = 10

        self.truck_matrix.matrix_view = np.zeros(self.user_class_dims[1])
        self.truck_matrix.matrix_view[self.truck_index[1], self.truck_index[2]] = 10

        self.moto_matrix.matrix_view = np.zeros(self.user_class_dims[2])
        self.moto_matrix.matrix_view[self.moto_index[1], self.moto_index[2]] = 10

        self.assignment.execute()
        df = self.assignment.results().reset_index(drop=False).fillna(0)
        print(df.columns)
        #df[['link_id', 'car_ab', 'trucks_ab', 'motorcycle_ab', 'PCE_AB', 'PCE_BA', 'PCE_tot']]

    # Basic tests for multi-class ODME
    # These are similar to the basic tests for single-class ODME
    # and are primarily just intended as sanity checks
    # They will follow the same structure - although they will test with different numbers
    # of classes.
    def test_all_zeros(self) -> None:
        """
        Check that running ODME on 3 user classes with all 0 demand matrices,
        returns 0 demand matrix when given a single count volume of 0 from each
        class.
        """
        # Set synthetic demand matrix & count volumes
        for i, matrix in enumerate(self.matrices):
            matrix.matrix_view = np.zeros(self.user_class_dims[i])
        
        count_volumes = pd.DataFrame(
            data=[[user_class, 1, 1, 0] for user_class in self.user_class_names],
            columns=self.count_vol_cols
        )

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Check for each class that the matrix is still 0's.
        for i, matrix in enumerate(self.matrices):
            np.testing.assert_allclose(
                matrix.matrix_view,
                np.zeros(self.user_class_dims[i])[:, :, np.newaxis],
                err_msg=f"The {self.user_class_names[i]} matrix was changed from 0 when initially a 0 matrix!"
            )

    def test_no_changes(self) -> None:
        """
        Check that running ODME on 3 user classes with original demand matrices
        and all count volumes corresponding to those currently assigned does not
        perturb original matrices.
        """
        # Get original flows:
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        for matrix in self.matrices:
            matrix.matrix_view = np.squeeze(matrix.matrix_view, axis=2)

        # Set the observed count volumes:
        flow = lambda i, matrix: assign_df.loc[assign_df["link_id"] == i, f"{matrix.view_names[0]}_ab"].values[0]
        count_volumes = pd.DataFrame(
            data=[
                [self.user_class_names[j], i, 1, flow(i, matrix)]
                for i in assign_df["link_id"]
                for j, matrix in enumerate(self.matrices)
                ],
            columns=self.count_vol_cols
        )

        # Store original matrices
        original_demands = [np.copy(matrix.matrix_view) for matrix in self.matrices]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Get results
        new_demands = odme.get_demands()

        # Check for each class that the matrix is still 0's.
        for i, matrices in enumerate(zip(original_demands, new_demands)):
            old, new = matrices
            np.testing.assert_allclose(
                old[:, :, np.newaxis],
                new,
                err_msg=f"The {self.user_class_names[i]} matrix was changed when given count volumes " +
                "which correspond to currently assigned volumes!"
            )

    # Will need a test checking everything works fine if you do not include count volumes for a particular class