import os
import uuid
import zipfile
from os.path import dirname
from os.path import join
from tempfile import gettempdir
from unittest import TestCase

import numpy as np
import pandas as pd

from aequilibrae import Graph
from aequilibrae import PathResults
from aequilibrae import Project
from aequilibrae import TrafficAssignment
from aequilibrae import TrafficClass
from utils.spatialite_utils import ensure_spatialite_binaries
from ...data import siouxfalls_project

TURNS_DF = pd.DataFrame([[1, 2, 6]], columns=["a", "b", "c"])


class TestTurnVolumes(TestCase):
    def setUp(self) -> None:
        ensure_spatialite_binaries()
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_path_files" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)
        print(proj_path)
        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix = self.project.matrices.get_matrix("demand_omx")
        print(self.matrix.get_matrix("matrix")[0][5])
        print(self.matrix.get_matrix("matrix")[2][5])
        self.matrix.computational_view()

        self.assignment = TrafficAssignment()
        self.assignclass = TrafficClass("car", self.car_graph, self.matrix)
        # self.assignclass_b = TrafficClass("other_car", self.car_graph, self.matrix)
        self.assignment.set_classes([self.assignclass])
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_multiple_link_sets(self):
        """
        Tests whether the Select Link feature works as wanted.
        Uses two examples: 2 links in one select link, and a single Selected Link
        Checks both the OD Matrix and Link Loading
        """
        self.assignment.set_algorithm("all-or-nothing")
        self.assignment.max_iter = 5
        self.assignment.rgap_target = 0.001
        self.assignment.set_save_path_files(True)
        self.assignment.execute()
        turning_movements = self.assignment.turning_volumes(TURNS_DF)
        print(turning_movements)
        print(self.assignment.results()[["matrix_ab", "matrix_ba"]])
