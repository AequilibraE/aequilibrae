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

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
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

        self.assignclass.set_select_links(
            [[(9, 1), (6, 1)],
             [(3, 1)]]
        )

        self.assignment.set_classes([self.assignclass])

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 2
        self.assignment.set_algorithm("msa")

        self.assignment.execute()
        for key, val in self.assignclass._aon_results.select_link_od.matrix.items():
            print(key, "shape: ", val.shape, " owns this matrix: ")
            print(val)
        print(self.assignclass.matrix.matrix_view)
        self.assertTrue(False)
        self.assertTrue(self.assignclass._sl_results is not None)
        # _sl_results.matricies == {(9, 1): AequilibraeMatrix(), (6, 1): AequilibraeMatrix()}
