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


class TestODMESetUp(TestCase):
    """
    Suite of Unit Tests for internal implementation of ODME class.
    Should not be ran during commits - only used for contrsuction purposes (ie implementation details can 
    change for internal functionality of ODME class).
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
        #self.odme_solver = ODME("car", self.car_graph, self.assignment, self.matrix, [10000])

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_playground(self) -> None:
        """
        Using this to figure out how API works
        Currently extracting the link flows corresponding to observed links following an execution
        """
        select_links = {"sl_9_1": [(9, 1)], "sl_6_0": [(6, 0)], "sl_4_1": [(4,1)]}
        self.assignclass.set_select_links(select_links)
        self.assignment.execute()
        sl_matrix = self.assignclass.results.select_link_od.matrix
        select_link_flow_df = self.assignment.select_link_flows().reset_index(drop=False).fillna(0)
        print(sl_matrix)
        #print(sl_matrix.keys())
        #print(select_link_flow_df)
        self.odme_solver = ODME(self.assignment, [((9,1), 10000)])
        #select_links = {"sl 6": [(6, 1)], "sl 3": [(3, 1)]}

        #self.assignclass.set_select_links(select_links)
        #self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        #print(assign_df)

        all_sl = []
        for sl in select_links.values():
            all_sl += sl
        col = {1: "matrix_ab", -1: "matrix_ba", 0: "matrix_tot"}
        obs_link_flows = []
        for sl in all_sl:
            obs_link_flows += [assign_df.loc[assign_df["link_id"] == sl[0], col[sl[1]]].values[0]]
        print(obs_link_flows)
