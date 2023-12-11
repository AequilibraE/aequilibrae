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

from aequilibrae import TrafficAssignment, TrafficClass, Graph
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project


class TestODMEMultiClassSetUp(TestCase):
    """
    Basic tests of ODME algorithm with multiple user classes.

    Currently taken from test_mc_traffic_assignment.py
    """

    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

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

        self.car_matrix = self.project.matrices.get_matrix("demand_mc")
        self.car_matrix.computational_view(["car"])

        self.truck_matrix = self.project.matrices.get_matrix("demand_mc")
        self.truck_matrix.computational_view(["trucks"])

        self.moto_matrix = self.project.matrices.get_matrix("demand_mc")
        self.moto_matrix.computational_view(["motorcycle"])

        self.assignment = TrafficAssignment()
        self.carclass = TrafficClass("car", self.car_graph, self.car_matrix)
        self.carclass.set_pce(1.0)
        self.motoclass = TrafficClass("motorcycle", self.moto_graph, self.moto_matrix)
        self.carclass.set_pce(0.2)
        self.truckclass = TrafficClass("truck", self.truck_graph, self.truck_matrix)
        self.carclass.set_pce(2.5)

        self.assignment.set_classes([self.carclass, self.truckclass, self.motoclass])

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 20
        self.assignment.set_algorithm("bfw")

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
        x = 1
