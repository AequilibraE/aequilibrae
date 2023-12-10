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
    Intended as both tests for optimisation and robustness of implementation.
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

        # Extra data specific to ODME:
        self.index = self.car_graph.nodes_to_indices
        self.dims = self.matrix.matrix_view.shape
        self.count_vol_cols = ["class", "link_id", "direction", "volume"]
        # Still need to add mode/class name to these!!!

        # Initial assignment parameters:
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
        # NOT YET IMPLEMENTED
        count_volumes = pd.DataFrame(
            data=[["car", 9, 1, 10000]],
            columns=self.count_vol_cols
        )
        odme_solver = ODME(self.assignment, count_volumes)
        demand_matrix = self.matrix.matrix_view
        #count_volumes = [10000]

        odme_solver.execute()
        new_demand_matrix = odme_solver.get_result()
        assert(np.sum(demand_matrix) - np.sum(new_demand_matrix) <= 10^-2) # Arbitrarily chosen value for now
        # Likely far too stringent of a condition.

    def test_bottom_left_zeroed_default(self) -> None:
        """
        Tests whether attempting to double the bottom left link value (link 37) only changes the O-D pairs
        13-12 & 24-12 (see QGIS for visualisation).
        
        First sets all demand to 0 (on all O-D pairs - both directions except for specified OD's), then assigns 
        and extracts flow on link 37 from 13-12 (ba direction). Attempts to perform ODME with observation of
        double flow on link 37.
        
        
        Checks whether resulting demand matrix only changes the expected cells.
        Asserts flow on  link is doubled.
        Expect convergence for such a simple case.
        """
        # Set synthetic demand matrix
        demand = np.zeros(self.matrix.matrix_view.shape)
        index = self.car_graph.nodes_to_indices
        demand[index[13], index[12]] = 1
        demand[index[24], index[12]] = 1
        self.matrix.matrix_view = demand

        # Extract assigned flow on link 37
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        old_flow = assign_df.loc[assign_df["link_id"] == 38, "matrix_ab"].values[0]

        # Perform ODME with doubled link flow on link 37
        # Execute with default options
        count_volumes = pd.DataFrame(
            data=[["car", 38, 1, 2 * old_flow]],
            columns=self.count_vol_cols
        )
        odme = ODME(self.assignment, count_volumes)
        odme.execute()
        new_demand = odme.get_result()

        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        new_flow = assign_df.loc[assign_df["link_id"] == 38, "matrix_ab"].values[0]

        # Assert link flow is in fact doubled:
        assert(new_flow == 2 * old_flow)
        
        # Assert only appropriate O-D pairs (13-12 & 24-12) have had demand changed
        od_13_12 = new_demand[index[13], index[12]]
        od_24_12 = new_demand[index[24], index[12]]
        assert np.sum(new_demand) == od_13_12 + od_24_12
        assert od_13_12 > 1 or od_24_12 > 1