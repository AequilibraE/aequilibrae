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
    Robust tests of ODME class with congested networks.
    This test suite should test both ODME with a single user class and multiple user classes.
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
        self.count_vol_cols = ["class", "link_id", "direction", "obs_volume"]
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
        self.assignment.max_iter = 5
        self.assignment.set_algorithm("msa")

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_playground(self) -> None:
        """
        Used for messing around with stuff and seeing how stuff works.

        Should be removed later.
        """
        self.assignclass.set_select_links({"sl_1_1": [(1, 1)]})
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        self.matrix.matrix_view = np.squeeze(self.matrix.matrix_view, axis=2)
        sl_matrix = self.assignclass.results.select_link_od.matrix["sl_1_1"].squeeze()
        sl_flows = self.assignment.select_link_flows().reset_index(drop=False).fillna(0)
        flow = assign_df.loc[assign_df["link_id"] == 1, "matrix_ab"].values[0]
        print(flow)
        calc = np.sum(sl_matrix)
        print(calc)

    def test_no_changes(self) -> None:
        """
        Checks ODME does nothing to a congested network when the observed volumes
        are equal to the initially assigned volumes.
        """
        # Get original flows:
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        # SQUISH EXTRA DIMENSION FOR NOW - DEAL WITH THIS PROPERLY LATER ON!!!
        self.matrix.matrix_view = np.squeeze(self.matrix.matrix_view, axis=2)

        # Set the observed count volumes:
        flow = lambda i: assign_df.loc[assign_df["link_id"] == i, "matrix_ab"].values[0]
        count_volumes = pd.DataFrame(
            data=[["car", i, 1, flow(i)] for i in assign_df["link_id"]],
            columns=self.count_vol_cols
        )

        # Store original matrix
        original_demand = np.copy(self.matrix.matrix_view)

        # Perform ODME:
        odme = ODME(self.assignment, count_volumes)
        odme.execute()
        new_demand, stats = odme.get_results()

        # Check results:
        np.testing.assert_allclose(
            original_demand,
            new_demand,
            err_msg="Original matrix was not obtained after perturbing slightly and running ODME!"
        )

    def test_all_volumes_given(self) -> None:
        """
        Takes original Sioux Falls network demand matrix and perturbs it slightly.
        Then executes ODME on this with all original flows from Sioux Falls network
        and the perturbed matrix.

        Checks we recover the original matrix.
        """
        # Get original flows:
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        # SQUISH EXTRA DIMENSION FOR NOW - DEAL WITH THIS PROPERLY LATER ON!!!
        self.matrix.matrix_view = np.squeeze(self.matrix.matrix_view, axis=2)

        # Set the observed count volumes:
        flow = lambda i: assign_df.loc[assign_df["link_id"] == i, "matrix_ab"].values[0]
        count_volumes = pd.DataFrame(
            data=[["car", i, 1, flow(i)] for i in assign_df["link_id"]],
            columns=self.count_vol_cols
        )

        # Store original matrix
        original_demand = np.copy(self.matrix.matrix_view)

        # Perturb original matrix:
        np.random.seed(0)
        perturbation_matrix = np.random.uniform(0.99, 1.01, size=self.dims)
        self.matrix.matrix_view = np.round(self.matrix.matrix_view * perturbation_matrix)

        # Perform ODME:
        odme = ODME(self.assignment, count_volumes, stop_crit=(300, 10, 10, 10))
        odme.execute()
        new_demand, stats = odme.get_results()
        odme.get_assignment_data().to_csv("/workspaces/aequilibrae/stats_all_vols.csv")

        # Check results:
        np.testing.assert_allclose(
            original_demand,
            new_demand,
            err_msg="Original matrix was not obtained after perturbing slightly and running ODME!"
        )

    def test_3_volumes_given(self) -> None:
        """
        Takes original Sioux Falls network demand matrix and perturbs it slightly.
        Then executes ODME on this with all original flows from Sioux Falls network
        and the perturbed matrix.

        Checks we recover the original matrix.

        TAKES ONLY 3 VOLUMES - NEED TO CLEAN UP TESTS LATER ON!!!
        """
        # Get original flows:
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        # SQUISH EXTRA DIMENSION FOR NOW - DEAL WITH THIS PROPERLY LATER ON!!!
        self.matrix.matrix_view = np.squeeze(self.matrix.matrix_view, axis=2)

        # Set the observed count volumes:
        flow = lambda i: assign_df.loc[assign_df["link_id"] == i, "matrix_ab"].values[0]
        count_volumes = pd.DataFrame(
            data=[["car", 1, 1, flow(1)], ["car", 30, 1, flow(30)], ["car", 52, 1, flow(52)]],
            columns=self.count_vol_cols
        )

        # Store original matrix
        original_demand = np.copy(self.matrix.matrix_view)

        # Perturb original matrix:
        np.random.seed(0)
        perturbation_matrix = np.random.uniform(0.99, 1.01, size=self.dims)
        self.matrix.matrix_view = np.round(self.matrix.matrix_view * perturbation_matrix)

        # Perform ODME:
        odme = ODME(self.assignment, count_volumes, stop_crit=(100, 10, 0.001, 1))
        odme.execute()
        new_demand, stats = odme.get_results()
        odme.get_assignment_data().to_csv("/workspaces/aequilibrae/stats_3_vols.csv")

        # Check results:
        np.testing.assert_allclose(
            original_demand,
            new_demand,
            err_msg="Original matrix was not obtained after perturbing slightly and running ODME!"
        )