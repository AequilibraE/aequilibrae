import os
import pathlib
import random
import sqlite3
import string
import uuid
from random import choice
from tempfile import gettempdir
from unittest import TestCase

import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, TrafficClass, Graph
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project


class TestTrafficAssignment(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_" + uuid.uuid4().hex)
        self.project = create_example(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)

        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        self.assignment = TrafficAssignment()
        self.assigclass = TrafficClass("car", self.car_graph, self.matrix)

        self.algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_matrix_with_wrong_type(self):
        self.matrix.matrix_view = np.array(self.matrix.matrix_view, np.int32)
        with self.assertRaises(TypeError):
            _ = TrafficClass("car", self.car_graph, self.matrix)

    def test_set_vdf(self):
        with self.assertRaises(ValueError):
            self.assignment.set_vdf("CQS")

        self.assignment.set_vdf("BPR")

    def test_set_classes(self):
        with self.assertRaises(AttributeError):
            self.assignment.set_classes([1, 2])

        with self.assertRaises(Exception):
            self.assignment.set_classes(self.assigclass)

        self.assignment.set_classes([self.assigclass])
        # self.fail()

    def test_algorithms_available(self):
        algs = self.assignment.algorithms_available()
        real = ["all-or-nothing", "msa", "frank-wolfe", "bfw", "cfw", "fw"]

        diff = [x for x in real if x not in algs]
        diff2 = [x for x in algs if x not in real]

        if len(diff) + len(diff2) > 0:
            self.fail("list of algorithms raised is wrong")

    def test_set_cores(self):
        with self.assertRaises(Exception):
            self.assignment.set_cores(3)

        self.assignment.add_class(self.assigclass)
        with self.assertRaises(ValueError):
            self.assignment.set_cores("q")

        self.assignment.set_cores(3)

    def test_set_algorithm(self):
        with self.assertRaises(AttributeError):
            self.assignment.set_algorithm("not an algo")

        self.assignment.add_class(self.assigclass)

        with self.assertRaises(Exception):
            self.assignment.set_algorithm("msa")

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 10

        for algo in self.algorithms:
            for _ in range(10):
                algo = "".join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])
                self.assignment.set_algorithm(algo)

        with self.assertRaises(AttributeError):
            self.assignment.set_algorithm("not a valid algorithm")

    def test_set_vdf_parameters(self):
        with self.assertRaises(Exception):
            self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_vdf("bpr")
        self.assignment.add_class(self.assigclass)
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

    def test_set_time_field(self):
        with self.assertRaises(ValueError):
            self.assignment.set_time_field("capacity")

        self.assignment.add_class(self.assigclass)

        N = random.randint(1, 50)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        with self.assertRaises(ValueError):
            self.assignment.set_time_field(val)

        self.assignment.set_time_field("free_flow_time")
        self.assertEqual(self.assignment.time_field, "free_flow_time")

    def test_set_capacity_field(self):
        with self.assertRaises(ValueError):
            self.assignment.set_capacity_field("capacity")

        self.assignment.add_class(self.assigclass)

        N = random.randint(1, 50)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        with self.assertRaises(ValueError):
            self.assignment.set_capacity_field(val)

        self.assignment.set_capacity_field("capacity")
        self.assertEqual(self.assignment.capacity_field, "capacity")

    def test_execute_and_save_results(self):
        conn = sqlite3.connect(os.path.join(siouxfalls_project, "project_database.sqlite"))
        results = pd.read_sql("select volume from links order by link_id", conn)

        self.assignment.add_class(self.assigclass)
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 10
        self.assignment.set_algorithm("msa")
        self.assignment.execute()

        msa10_rgap = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume.values)[0, 1]
        self.assertLess(0.8, correl)

        self.assignment.max_iter = 500
        self.assignment.rgap_target = 0.001
        self.assignment.set_algorithm("msa")
        self.assignment.execute()
        msa25_rgap = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.98, correl)

        self.assignment.set_algorithm("frank-wolfe")
        self.assignment.execute()

        fw25_rgap = self.assignment.assignment.rgap
        fw25_iters = self.assignment.assignment.iter

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.99, correl)

        self.assignment.set_algorithm("cfw")
        self.assignment.execute()
        cfw25_rgap = self.assignment.assignment.rgap
        cfw25_iters = self.assignment.assignment.iter

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.995, correl)

        # For the last algorithm, we set skimming
        self.car_graph.set_skimming(["free_flow_time", "distance"])
        assigclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assignment.set_classes([assigclass])

        self.assignment.set_algorithm("bfw")
        self.assignment.execute()
        bfw25_rgap = self.assignment.assignment.rgap
        bfw25_iters = self.assignment.assignment.iter

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.999, correl)

        self.assertLess(msa25_rgap, msa10_rgap)
        # MSA and FW do not reach 1e-4 within 500 iterations, cfw and bfw do
        self.assertLess(fw25_rgap, msa25_rgap)
        self.assertLess(cfw25_rgap, self.assignment.rgap_target)
        self.assertLess(bfw25_rgap, self.assignment.rgap_target)
        # we expect bfw to converge quicker than cfw
        self.assertLess(cfw25_iters, fw25_iters)
        self.assertLess(bfw25_iters, cfw25_iters)

        self.assignment.save_results("save_to_database")
        self.assignment.save_skims(matrix_name="all_skims", which_ones="all")

        with self.assertRaises(ValueError):
            self.assignment.save_results("save_to_database")

    def test_info(self):
        iterations = random.randint(1, 10000)
        rgap = random.random() / 10000
        algo = choice(self.algorithms)

        self.assignment.add_class(self.assigclass)
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = iterations
        self.assignment.rgap_target = rgap
        self.assignment.set_algorithm(algo)

        # TY
        for _ in range(10):
            algo = "".join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])

        dct = self.assignment.info()
        if algo.lower() == "fw":
            algo = "frank-wolfe"
        self.assertEqual(dct["Algorithm"], algo.lower(), "Algorithm not correct in info method")

        self.assertEqual(dct["Maximum iterations"], iterations, "maximum iterations not correct in info method")
