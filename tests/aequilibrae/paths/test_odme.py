import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, ODME
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
        new_demand = odme.get_demands()[0]

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
        algorithm = "reg_spiess"

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
        perturbation = 5 # %
        perturbation_matrix = np.random.uniform(1 - perturbation/100, 1 + perturbation/100, size=self.dims)
        self.matrix.matrix_view = np.round(self.matrix.matrix_view * perturbation_matrix)

        # Perform ODME:
        odme = ODME(self.assignment,
            count_volumes,
            stop_crit={"max_outer": 10, "max_inner": 500, "convergence_crit": 1, "inner_convergence": 0.1},
            alpha=0.24,
            algorithm=algorithm)
        #x = odme.estimate_alpha(0.1)
        odme.execute()
        new_demand = odme.get_demands()[0]
        odme.get_all_statistics().to_csv(f"/workspaces/aequilibrae/odme_stats/stats_all_vols_{algorithm}.csv")
        odme.get_iteration_factors().to_csv(f"/workspaces/aequilibrae/odme_stats/stats_all_factors_{algorithm}.csv")
        odme.get_cumulative_factors().to_csv(f"/workspaces/aequilibrae/odme_stats/stats_cumulative_factors_{algorithm}.csv")
        pd.DataFrame({"Demands": new_demand.ravel()}).to_csv(f"/workspaces/aequilibrae/odme_stats/new_demand_matrix_{algorithm}.csv")

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
        odme = ODME(self.assignment,
            count_volumes,
            stop_crit={"max_outer": 100,
                "max_inner": 100,
                "convergence_crit": 0.00001,
                "inner_convergence": 0.00001},
            algorithm="spiess")
        odme.execute()
        new_demand = odme.get_demands()[0]
        odme.get_all_statistics().to_csv("/workspaces/aequilibrae/odme_stats/stats_3_vols.csv")
        odme.get_iteration_factors().to_csv("/workspaces/aequilibrae/odme_stats/stats_3_factors.csv")

        # Check results:
        np.testing.assert_allclose(
            original_demand,
            new_demand,
            err_msg="Original matrix was not obtained after perturbing slightly and running ODME!"
        )