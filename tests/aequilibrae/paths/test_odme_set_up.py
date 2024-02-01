"""Basic tests for ODME infrastructure."""

import os
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase
import pandas as pd
import numpy as np

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, ODME
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project


class TestODMESingleClassSetUp(TestCase):
    """
    Basic unit tests for ODME single class execution
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
        self.matrix = self.project.matrices.get_matrix("demand_aem")
        self.matrix.computational_view()

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

        # Extra data for convenience:
        self.index = self.car_graph.nodes_to_indices
        # Extend dimensions by 1 to enable an AequilibraeMatrix to be used
        self.dims = self.matrix.matrix_view.shape + (1,)

        # Change this to test different algorithms
        self.algorithm = "spiess"

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    # 1) Edge Cases
    def test_basic_1_1(self) -> None:
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        single count volume of 0.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrices = np.zeros(self.dims)
        count_volumes = pd.DataFrame(
            data=[["car", 1, 1, 0]],
            columns=ODME.COUNT_VOLUME_COLS
        )

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Check result:
        np.testing.assert_allclose(
                np.zeros(self.dims),
                odme.get_demands()[0],
                err_msg="0 demand matrix with single count volume of 0 does not return 0 matrix",
        )

    def test_basic_1_2(self) -> None:
        """
        Given a demand matrix with 0 demand at certain OD's,
        following ODME the new demand matrix should have 0 demand at those OD's
        with many count volumes.
        """
        # Set synthetic demand matrix & count volumes
        demand = np.ones(self.dims)
        zeroes = [(18, 6), (5, 11), (11, 5), (23, 2), (13, 19), (19, 21), (19, 24), (17, 5)]
        for o, d in zeroes:
            demand[self.index[o], self.index[d], 0] = 0
        self.matrix.matrices = demand

        data = [
            ["car", 9, 1, 30],
            ["car", 11, 1, 2500],
            ["car", 35, 1, 0],
            ["car", 18, 1, 100],
            ["car", 6, 1, 2],
            ["car", 65, 1, 85],
            ["car", 23, 1, 0]
        ]
        count_volumes = pd.DataFrame(data=data, columns=ODME.COUNT_VOLUME_COLS)

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()
     
        # Check result:
        err_msg = "Demand matrix with many 0 entries, has non-zero demand following ODME at one of those entries"
        for o, d in zeroes:
            np.testing.assert_array_equal(
                odme.get_demands()[0][self.index[o], self.index[d], 0],
                0,
                err_msg=err_msg,
            )

    def test_basic_1_3(self) -> None:
        """
        Given count volumes which are identical to the assigned volumes of an 
        initial demand matrix - ODME should not change this demand matrix (since
        we are looking for a local solution and this already provides one).
        
        Also checks that the shape of the resulting matrix matches the intial
        demand matrix.

        Checks with many count volumes.
        """
        init_demand = np.copy(self.matrix.matrix_view)

        # Extract assigned flow on various links
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        links = [1,2,4,5,6,8,11,12,14,19,23,26,32,38,49,52,64,71,72]
        flows = [assign_df.loc[assign_df["link_id"] == link, "matrix_ab"].values[0]
            for link in links]

        # Perform ODME with unchanged count volumes
        count_volumes = pd.DataFrame(
            data=[["car", link, 1, flows[i]] for i, link in enumerate(links)],
            columns=ODME.COUNT_VOLUME_COLS
        )
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Check results
        np.testing.assert_allclose(
            init_demand[:, :, np.newaxis],
            odme.get_demands()[0],
            err_msg="Demand matrix changed when given many links with observed volume equal to initial assigned volumes"
        )

    # 2) Input Validity
    def test_basic_2_1(self) -> None:
        """
        Check ValueError is raised if count volumes are not given appropriately.
        These include:
        - no count volumes given
        - negative count volumes given
        - duplicate count volumes given
        - non-float/integer count volumes given
        """
        # No count volumes:
        with self.assertRaises(ValueError):
            ODME(self.assignment,
                pd.DataFrame(data=[], columns=ODME.COUNT_VOLUME_COLS),
                algorithm=self.algorithm)

        # Negative count volumes:
        links = [1, 3, 10, 30, 36, 41, 49, 57, 62, 66, 69, 70]
        count_volumes = pd.DataFrame(
            data=[["car", link, 1, -link] for link in links],
            columns=ODME.COUNT_VOLUME_COLS
        )
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, algorithm=self.algorithm)

        # Duplicate count volumes:
        count_volumes = pd.DataFrame(
            data=[["car", 1, 1, i] for i in range(5)],
            columns=ODME.COUNT_VOLUME_COLS
        )
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, algorithm=self.algorithm)

        # Non-float/integer count volumes:
        count_volumes = pd.DataFrame(
            data=[["car", 1, 1, '7'], ["car", 10, 1, [1]], ["car", 15, 1, (1, 2)]],
            columns=ODME.COUNT_VOLUME_COLS
        )
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, algorithm=self.algorithm)

    def test_basic_2_2(self) -> None:
        """
        Check ValueError is raised if invalid stopping criteria are given
        or stopping criteria are given with missing criteria.
        """
        count_volumes = pd.DataFrame(
            data=[["car", 1, 1, 1]],
            columns=ODME.COUNT_VOLUME_COLS
        )

        # Check invalid (0) max iterations
        stop_crit = {"max_outer": 0,
            "max_inner": 0,
            "convergence_crit": 10**-4,
            "inner_convergence": 10**-4
            }
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, stop_crit=stop_crit, algorithm=self.algorithm)

        # Check invalid (negative) convergence
        stop_crit = {"max_outer": 10,
            "max_inner": 10,
            "convergence_crit": -10**-4,
            "inner_convergence": -10**-4
            }
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, stop_crit=stop_crit, algorithm=self.algorithm)

        # Check missing criteria
        stop_crit = {"max_outer": 10,
            "max_inner": 10,
            "convergence_crit": 10**-4,
            }
        with self.assertRaises(ValueError):
            ODME(self.assignment, count_volumes, stop_crit=stop_crit, algorithm=self.algorithm)

    # Simple Test Cases (Exact Results Expected For Spiess):
    def test_basic_3_1(self) -> None:
        """
        Test for single count volume observation which is double the currently assigned flow when 
        setting only 2 OD pairs to be non-negative such that all flow enters this link.
        NOTE - we are using small flows with little congestion.

        Link is 38, and OD pairs are 13-12 & 24-12 (bottom left area of graph in QGIS)

        Check that the new flow following ODME matches the observed flow.
        """
        # Set synthetic demand matrix
        demand = np.zeros(self.dims)
        demand[self.index[13], self.index[12], 0] = 1
        demand[self.index[24], self.index[12], 0] = 1
        self.matrix.matrices = demand

        # Extract assigned flow on link 38
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        old_flow = assign_df.loc[assign_df["link_id"] == 38, "matrix_ab"].values[0]

        # Perform ODME with doubled link flow on link 38
        count_volumes = pd.DataFrame(
            data=[["car", 38, 1, 2 * old_flow]],
            columns=ODME.COUNT_VOLUME_COLS
        )
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Get results
        new_demand = odme.get_demands()[0]
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        new_flow = assign_df.loc[assign_df["link_id"] == 38, "matrix_ab"].values[0]

        # Assert link flow is doubled:
        self.assertAlmostEqual(new_flow, 2 * old_flow)    

        # Assert only appropriate O-D's have increased non-zero demand
        od_13_12 = new_demand[self.index[13], self.index[12]]
        od_24_12 = new_demand[self.index[24], self.index[12]]
        self.assertAlmostEqual(np.sum(new_demand), od_13_12 + od_24_12)
        self.assertTrue(od_13_12 > 1 or od_24_12 > 1)

    def test_basic_3_2(self) -> None:
        """
        Test for two count volume observations with competing priorities for links (ie differing 
        observed volumes). Only has 1 non-zero OD pair which influences both links.

        Links are 5 & 35, and OD pair is 13-1 (see left side of graph in QGIS)

        Check that error from assigned volumes and observed volumes is balanced across both links.
        We expect flow on 5 and 35 to be equal and between the count volume on each
        """
        # Set synthetic demand matrix
        demand = np.zeros(self.dims)
        demand[self.index[13], self.index[1], 0] = 10
        self.matrix.matrices = demand

        # Perform ODME with competing link flows on 5 & 35
        count_volumes = pd.DataFrame(
            data=[["car", 5, 1, 100], ["car", 35, 1, 50]],
            columns=ODME.COUNT_VOLUME_COLS
        )
        odme = ODME(self.assignment, count_volumes, algorithm=self.algorithm)
        odme.execute()

        # Get Results:
        new_demand = odme.get_demands()[0]
        self.assignment.execute()
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        flow_5 = assign_df.loc[assign_df["link_id"] == 5, "matrix_ab"].values[0]
        flow_35 = assign_df.loc[assign_df["link_id"] == 35, "matrix_ab"].values[0]

        # Assert link flows are equal:
        self.assertAlmostEqual(flow_5, flow_35,
            msg=f"Expected balanced flows but are: {flow_5} and {flow_35}")

        # Assert link flows are balanced halfway between each other:
        self.assertTrue(flow_5 > 50 and flow_5 < 100,
            msg="Expected flows to be halfway between 50 & 100")

        # Assert only appropriate O-D's have had demand changed
        od_13_1 = new_demand[self.index[13], self.index[1]]
        self.assertAlmostEqual(np.sum(new_demand), od_13_1,
            msg="Unexpected OD pair has non-zero demand")

class TestODMEMultiClassSetUp(TestCase):
    """
    Basic unit tests for ODME multiple class execution

    NOTE - we use the data stored for test_mc_traffic_assignment.py
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

    # Basic tests for multi-class ODME
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
