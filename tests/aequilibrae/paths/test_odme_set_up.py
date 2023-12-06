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

        self.index = self.car_graph.nodes_to_indices
        self.dims = self.matrix.matrix_view.shape

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


    # Basic tests check basic edge cases, invalid inputs and a few simple inputs:
    # 1) Edge Cases
    # 2) Input Validity Checking (Ensuring API is Consistent)
    # 3) General Test Cases (Using Synthetic Demand Matrices & Pre-determined Results)

    # 1) Edge Cases
    def test_basic_1_1_a(self) -> None: 
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        single count volume of 0.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((1, 1), 0)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with single count volume of 0 does not return 0 matrix",
        )

    def test_basic_1_1_b(self) -> None: 
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        two count volumes of 0.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((1, 1), 0), ((5, 1), 0)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with 2 count volumes of 0 does not return 0 matrix",
        )

    def test_basic_1_1_c(self) -> None: 
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        many count volumes of 0.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((i, 1), 0) for i in range(1, 30, 2)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with many count volumes of 0 does not return 0 matrix",
        )

    def test_basic_1_1_d(self) -> None:
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        single non-zero count volume.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((1, 1), 10)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with single non-zero count volume does not return 0 matrix",
        )

    def test_basic_1_1_e(self) -> None: 
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        two non-zero count volumes.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((1, 1), 10), ((2, 1), 30)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with two non-zero count volumes does not return 0 matrix",
        )

    def test_basic_1_1_f(self) -> None: 
        """
        Check that running ODME with 0 demand matrix returns 0 matrix, with
        many non-zero count volumes.
        """
        # Set synthetic demand matrix & count volumes
        self.matrix.matrix_view = np.zeros(self.matrix.matrix_view.shape)
        count_volumes = [((i, 1), (i * 30) % ((i + 3) % 7)) for i in range(2, 30, 2)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_allclose(
                np.zeros(self.matrix.matrix_view.shape),
                odme.get_result(),
                err_msg="0 demand matrix with many non-zero count volumes does not return 0 matrix",
        )

    def test_basic_1_2_a(self) -> None:
        """
        Given a demand matrix with 0 demand at a single OD pair,
        following ODME the new demand matrix should have 0 demand at that OD pair
        with single count volume.
        """
        # Set synthetic demand matrix & count volumes
        demand = np.ones(self.matrix.matrix_view.shape)
        demand[self.index[13], self.index[12]] = 0
        self.matrix.matrix_view = demand

        count_volumes = [((9, 1), 30)]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()

        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_array_equal(
            odme.get_result()[self.index[13], self.index[12]],
            0,
            err_msg="Demand matrix with single 0 at OD 13-12, has non-zero demand following ODME",
        )

    def test_basic_1_2_b(self) -> None:
        """
        Given a demand matrix with 0 demand at a single OD pair,
        following ODME the new demand matrix should have 0 demand at that OD pair
        with many count volumes.
        """
        # Set synthetic demand matrix & count volumes
        demand = np.ones(self.matrix.matrix_view.shape)
        demand[self.index[18], self.index[6]] = 0
        self.matrix.matrix_view = demand

        count_volumes = [
            ((9, 1), 30),
            ((11, 1), 25),
            ((35, 1), 0),
            ((18, 1), 100),
            ((6, 1), 2),
            ((65, 1), 85),
            ((23, 1), 0)
        ]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()
     
        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        np.testing.assert_array_equal(
            odme.get_result()[self.index[13], self.index[12]],
            0,
            err_msg="Demand matrix with single 0 at OD 18-6, has non-zero demand following ODME",
        )

    def test_basic_1_2_c(self) -> None:
        """
        Given a demand matrix with 0 demand at many OD pairs,
        following ODME the new demand matrix should have 0 demand at those OD pair
        with single count volume.
        """
        # Set synthetic demand matrix & count volumes
        demand = np.ones(self.matrix.matrix_view.shape)
        zeroes = [(18, 6), (5, 11), (11, 5), (23, 2), (13, 19), (19, 21), (19, 24), (17, 5)]
        for o, d in zeroes:
            demand[self.index[o], self.index[d]] = 0
        self.matrix.matrix_view = demand

        count_volumes = [
            ((9, 1), 30),
        ]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()
     
        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        err_msg = "Demand matrix with many 0 entries, has non-zero demand following ODME at one of those entries"
        for o, d in zeroes:
            np.testing.assert_array_equal(
                odme.get_result()[self.index[o], self.index[d]],
                0,
                err_msg=err_msg,
            )
        
    def test_basic_1_2_d(self) -> None:
        """
        Given a demand matrix with 0 demand at many OD pairs,
        following ODME the new demand matrix should have 0 demand at those OD pair
        with many count volumes.
        """
        # Set synthetic demand matrix & count volumes
        demand = np.ones(self.matrix.matrix_view.shape)
        zeroes = [(18, 6), (5, 11), (11, 5), (23, 2), (13, 19), (19, 21), (19, 24), (17, 5)]
        for o, d in zeroes:
            demand[self.index[o], self.index[d]] = 0
        self.matrix.matrix_view = demand

        count_volumes = [
            ((9, 1), 30),
            ((11, 1), 2500),
            ((35, 1), 0),
            ((18, 1), 100),
            ((6, 1), 2),
            ((65, 1), 85),
            ((23, 1), 0)
        ]

        # Run ODME algorithm.
        odme = ODME(self.assignment, count_volumes)
        odme.execute()
     
        # Check result:
        # SHOULD I BE TESTING EXACTNESS HERE? IE. USE SOMETHING OTHER THAN allclose??
        err_msg = "Demand matrix with many 0 entries, has non-zero demand following ODME at one of those entries"
        for o, d in zeroes:
            np.testing.assert_array_equal(
                odme.get_result()[self.index[0], self.index[d]],
                0,
                err_msg=err_msg,
            )

    def test_basic_1_3(self) -> None:
        """
        Given count volumes which are identical to the assigned volumes of an 
        initial demand matrix - ODME should not change this demand matrix (since
        we are looking for a local solution and this already provides one).
        """
        assert False
    
    def test_basic_1_4(self) -> None:
        """
        Check that the shape of the resulting matrix following ODME is the same as the
        shape of the initial demand matrix.
        
        Split into sections a,b,c...
        """
        assert False

    # 2) Input Validity
    def test_basic_2_1_a(self) -> None:
        """
        Check that the ODME class does not accept input with no count volumes.
        Current API raises (DECIDE WHICH TYPE OF) error in this case.

        (NOTE - this is specific to this API, we could choose to simply return
        the initial demand matrix with no perturbation).
        """
        assert False

    def test_basic_2_1_b(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if negative count volumes are given.
        """
        assert False

    def test_basic_2_1_c(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if multiple count volumes
        are given for the same link.
        """
        assert False

    def test_basic_2_1_d(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if a given link does not exist.

        (NOTE - this may be a case of COUPLING since this error may be propagated from
        the TrafficClass class).
        """
        assert False

    def test_basic_2_2_a(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if input assignment object 
        has no classes set.
        """
        assert False

    def test_basic_2_2_b(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if input assignment object 
        has no volume delay function set.
        """
        assert False

    def test_basic_2_2_c(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if input assignment object 
        has no assignment algorithm set.
        """
        assert False

    def test_basic_2_3(self) -> None:
        """
        Check (DECIDE WHICH TYPE OF) error is raised if input demand matrix contains 
        negative values.

        (NOTE - this may be intended to be a part of some other part of the API and shouldn't
        be tested here).
        """
        assert False

    # Simple Test Cases:
    def test_basic_3_1(self) -> None:
        """
        Input single count volume representing OD path from node i to node j
        along link a, with demand matrix which is 0 everywhere except on OD pair (i, j).
        Ensure count volume is small enough that the best (i, j) path is via link a.

        Check that the only OD pair that has changed is at index (i, j) and that
        the assignment produces the required flow.
        """
        assert False

    # Add test with multiple classes
