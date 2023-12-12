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

        self.truck_matrix = self.project.matrices.get_matrix("demand_mc")
        self.truck_matrix.computational_view(["trucks"])

        self.moto_matrix = self.project.matrices.get_matrix("demand_mc")
        self.moto_matrix.computational_view(["motorcycle"])

        # Create assignment object and assign classes
        self.assignment = TrafficAssignment()
        self.carclass = TrafficClass("car", self.car_graph, self.car_matrix)
        self.carclass.set_pce(1.0)
        self.motoclass = TrafficClass("motorcycle", self.moto_graph, self.moto_matrix)
        self.carclass.set_pce(0.2)
        self.truckclass = TrafficClass("truck", self.truck_graph, self.truck_matrix)
        self.carclass.set_pce(2.5)

        self.assignment.set_classes([self.carclass, self.truckclass, self.motoclass])

        # Set assignment parameters
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 20
        self.assignment.set_algorithm("bfw")

        # Store parameters needed for ODME/demand matrix manipulation:
        self.count_vol_cols = ["class", "link_id", "direction", "obs_volume"]
        self.car_index = self.car_graph.nodes_to_indices
        self.car_dims = self.car_matrix.matrix_view.shape
        self.truck_index = self.truck_graph.nodes_to_indices
        self.truck_dims = self.truck_matrix.matrix_view.shape
        self.moto_index = self.moto_graph.nodes_to_indices
        self.moto_dims = self.moto_matrix.matrix_view.shape

        self.user_class_names = ["car", "motorcycle", "truck"]
        self.user_classes = [self.carclass, self.motoclass, self.truckclass]
        self.user_class_dims = [self.car_dims, self.moto_dims, self.truck_dims]
        self.matrices = [self.car_matrix, self.moto_matrix, self.truck_matrix]

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
        self.assignment.execute()   
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        #print("Assignment Columns: ", assign_df.columns)
        print(assign_df[["link_id", "car_ab", "trucks_ab", "motorcycle_ab"]].head())

    # Basic tests for multi-class ODME
    # These are similar to the basic tests for single-class ODME
    # and are primarily just intended as sanity checks
    # They will follow the same structure - although they will test with different numbers
    # of classes.
    def test_basic_1_1_a(self) -> None:
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
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        for i, matrix in enumerate(self.matrices):
            np.testing.assert_allclose(
                matrix.matrix_view,
                np.zeros(self.user_class_dims[i]),
                err_msg=f"The {self.user_class_names[i]} matrix was changed from 0 when initially a 0 matrix!"
            )
