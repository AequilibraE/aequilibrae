import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, ODME
from aequilibrae.matrix import AequilibraeMatrix
from ...data import siouxfalls_project


class TestODME(TestCase):
    """
    Tests final outputs of ODME execution with various inputs.
    Runs accuracy and time tests.
    """

    def setUp(self) -> None:
        # Set up data:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        proj_path = os.path.join(gettempdir(), "test_odme_files" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)

        # Initialise project:
        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
    
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        # Initial assignment:
        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assignment.set_classes([self.assignclass])
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")
        self.assignment.max_iter = 1
        self.assignment.set_algorithm("bfw")

        # Set up ODME solver with default stopping conditions: 
        # NEEDS TO BE CHANGED - SHOULD BE CREATED WITHIN INDIVIDUAL TESTS
        # Final value needs to be a data frame
        self.odme_solver = ODME(self.assignment, [((9,1), 10000)])

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()    

    def test_single_observation(self) -> None:
        """
        Test whether the ODME algorithm works correctly when the set of observed links is a single link.
        Run:
        >>> new_demand_matrix = ODME.execute(demand_matrix, {[9,1]: 10000}) 

        We expect the following to hold:
        - When assigning demand_matrix we expect link volume on [9,1] to be 9000
        - sum(demand_matrix) ~= sum(new_demand_matrix)
        - When assigning new_demand_matrix we expect link volume on [9, 1] to be close to 100000
        """
        demand_matrix = self.matrix
        #count_volumes = [10000]

        new_demand_matrix = self.odme_solver.execute()
        assert(np.sum(demand_matrix) - np.sum(new_demand_matrix) <= 10^-2) # Arbitrarily chosen value for now
        #computational_view()