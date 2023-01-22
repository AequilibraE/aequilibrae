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

        self.algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_select_link_results(self):
        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)

        self.assignclass.set_select_links([[(9, 1), (6, 1)], [(3, 1)]])

        self.assignment.set_classes([self.assignclass])

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        od_mask = create_od_mask(self.assignclass.matrix.matrix_view, self.assignclass.graph, 3)
        print(od_mask)
        self.assertTrue(False)
        self.assignment.max_iter = 2
        self.assignment.set_algorithm("msa")
        self.assignment.set_cores(1)
        self.assignment.execute()
        self.assertTrue(False)
        self.assertTrue(self.assignclass._sl_results is not None)
        # _sl_results.matricies == {(9, 1): AequilibraeMatrix(), (6, 1): AequilibraeMatrix()}


def create_od_mask(demand, graph, sl):
    res = PathResults()
    # This uses the UNCOMPRESSED graph, since we don't know which nodes the user may ask for
    graph.set_graph("free_flow_time")
    res.prepare(graph)

    a = []
    # compute a path from node 8 to 13
    for origin in range(1, 24):
        b=[]
        for dest in range(1, 24):
            # print(dest)
            if origin == dest:
                pass
            else:
                res.compute_path(origin, dest)
            # print(res.path_nodes)
            if res.path_nodes is not None:
                b.append(list(res.path_nodes))
        a.append(b)
    # print(a)
    node_pair = graph.graph.iloc[sl-1]["a_node"]+1, graph.graph.iloc[sl-1]["b_node"]+1
    print(node_pair)
    mask = dict()
    for origin, val in enumerate(a):
        for dest, path in enumerate(val):
            for k in range(len(path)):
                if origin == dest:
                    pass
                elif path[k] == node_pair[1] and path[k-1] == node_pair[0]:
                    mask[(origin, dest)] = True
    print(mask)
    sl_od = np.zeros((24, 24))
    for origin in range(24):
        for dest in range(24):
            if mask.get((origin, dest)):
                sl_od[origin, dest] = demand[origin, dest]
    return sl_od