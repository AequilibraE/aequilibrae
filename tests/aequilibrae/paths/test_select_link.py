import os
import pathlib
import uuid
import zipfile
from os.path import join, dirname
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase
import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, PathResults
from ...data import siouxfalls_project


class TestSelectLink(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_path_files" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)
        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assignment.set_classes([self.assignclass])
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")
        self.assignment.max_iter = 1
        self.assignment.set_algorithm("msa")

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_multiple_link_sets(self):
        """
        Tests whether the Select Link feature works as wanted.
        Uses two examples: 2 links in one select link, and a single Selected Link
        Checks both the OD Matrix and Link Loading
        """
        self.assignclass.set_select_links({"9 or 6": [(9, 1), (6, 1)], "just 3": [(3, 1)], "5 for fun": [(5, 1)]})
        self.assignment.execute()
        for key in self.assignclass._selected_links.keys():
            od_mask, link_loading = create_od_mask(
                self.assignclass.matrix.matrix_view, self.assignclass.graph, self.assignclass._selected_links[key]
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_od.matrix[key][:, :, 0],
                od_mask,
                err_msg="OD SL matrix for: " + str(key) + " does not match",
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_loading[key],
                link_loading,
                err_msg="Link loading SL matrix for: " + str(key) + " does not match",
            )

    def test_equals_demand_one_origin(self):
        """
        Test to ensure the Select Link functionality behaves as required.
        Tests to make sure the OD matrix works when all links surrounding one origin are selected
        Confirms the Link Loading is done correctly in this case
        """
        self.assignclass.set_select_links({"1, 4, 3, and 2": [(1, 1), (4, 1), (3, 1), (2, 1)]})

        self.assignment.execute()

        for key in self.assignclass._selected_links.keys():
            od_mask, link_loading = create_od_mask(
                self.assignclass.matrix.matrix_view, self.assignclass.graph, self.assignclass._selected_links[key]
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_od.matrix[key][:, :, 0],
                od_mask,
                err_msg="OD SL matrix for: " + str(key) + " does not match",
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_loading[key],
                link_loading,
                err_msg="Link loading SL matrix for: " + str(key) + " does not match",
            )

    def test_single_demand(self):
        """
        Tests the functionality of Select Link when given a custom demand matrix, where only 1 OD pair has demand on it
        Confirms the OD matrix behaves, and the Link Loading is just on the path of this OD pair
        """
        custom_demand = np.zeros((24, 24, 1))
        custom_demand[0, 23, 0] = 1000
        self.matrix.matrix_view = custom_demand
        self.assignclass.matrix = self.matrix

        self.assignclass.set_select_links({"39, 66, or 73": [(39, 1), (66, 1), (73, 1)]})

        self.assignment.execute()
        for key in self.assignclass._selected_links.keys():
            od_mask, link_loading = create_od_mask(
                self.assignclass.matrix.matrix_view, self.assignclass.graph, self.assignclass._selected_links[key]
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_od.matrix[key][:, :, 0],
                od_mask,
                err_msg="OD SL matrix for: " + str(key) + " does not match",
            )
            np.testing.assert_allclose(
                self.assignclass.results.select_link_loading[key],
                link_loading,
                err_msg="Link loading SL matrix for: " + str(key) + " does not match",
            )

    def test_select_link_network_loading(self):
        """
        Test to ensure the SL_network_loading method correctly does the network loading
        """
        self.assignment.execute()
        non_sl_loads = self.assignclass.results.get_load_results()
        self.setUp()
        self.assignclass.set_select_links({"39, 66, or 73": [(39, 1), (66, 1), (73, 1)]})
        self.assignment.execute()
        sl_loads = self.assignclass.results.get_load_results()
        np.testing.assert_allclose(non_sl_loads.matrix_tot, sl_loads.matrix_tot)

    def test_duplicate_links(self):
        """
        Tests to make sure the user api correctly filters out duplicate links in the compressed graph
        """
        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assignclass.set_select_links({"test": [(1, 1), (1, 1)]})
        self.assertEquals(len(self.assignclass._selected_links["test"]), 1, "Did not correctly remove duplicate link")

    def test_link_out_of_bounds(self):
        """
        Test to confirm the user api correctly identifies when an input node is invalid for the current graph
        """
        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assertRaises(ValueError, self.assignclass.set_select_links, {"test": [(78, 1), (1, 1)]})


def create_od_mask(demand: np.array, graph: Graph, sl):
    res = PathResults()
    # This uses the UNCOMPRESSED graph, since we don't know which nodes the user may ask for
    graph.set_graph("free_flow_time")
    res.prepare(graph)

    def g(o, d):
        res.compute_path(o, d)
        return list(res.path_nodes) if (res.path_nodes is not None and o != d) else []

    a = [[g(o, d) for d in range(1, 25)] for o in range(1, 25)]
    sl_links = []
    for i in range(len(sl)):
        node_pair = graph.graph.iloc[sl[i]]["a_node"] + 1, graph.graph.iloc[sl[i]]["b_node"] + 1
        sl_links.append(node_pair)
    mask = dict()
    for origin, val in enumerate(a):
        for dest, path in enumerate(val):
            for k in range(1, len(path)):
                if origin == dest:
                    pass
                elif (path[k - 1], path[k]) in sl_links:
                    mask[(origin, dest)] = True
    sl_od = np.zeros((24, 24))
    for origin in range(24):
        for dest in range(24):
            if mask.get((origin, dest)):
                sl_od[origin, dest] = demand[origin, dest]

    # make link loading
    loading = np.zeros((76, 1))
    for orig, dest in mask.keys():
        path = a[orig][dest]
        for i in range(len(path) - 1):
            link = (
                graph.graph[(graph.graph["a_node"] == path[i] - 1) & (graph.graph["b_node"] == path[i + 1] - 1)][
                    "link_id"
                ].values[0]
                - 1
            )
            loading[link] += demand[orig, dest]
    return sl_od, loading
