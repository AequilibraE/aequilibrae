from unittest import TestCase
import os
from shutil import copytree
import uuid
import string
import random
from random import choice
from tempfile import gettempdir
import numpy as np
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project import Project
from aequilibrae import TrafficAssignment, TrafficClass, Graph

from ...data import siouxfalls_project, siouxfalls_demand, data_folder


class TestTrafficAssignment(TestCase):
    def setUp(self) -> None:
        os.environ['PATH'] = os.path.join(gettempdir(), 'temp_data') + ';' + os.environ['PATH']

        self.matrix = AequilibraeMatrix()
        self.matrix.load(siouxfalls_demand)
        self.matrix.computational_view()

        proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, proj_dir)

        self.project = Project()
        self.project.open(proj_dir)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs['c']  # type: Graph
        self.car_graph.set_graph('free_flow_time')
        self.car_graph.set_blocked_centroid_flows(False)

        self.assignment = TrafficAssignment()
        self.assigclass = TrafficClass(self.car_graph, self.matrix)

        self.algorithms = ['msa', 'cfw', 'bfw', 'frank-wolfe']

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_set_vdf(self):
        with self.assertRaises(ValueError):
            self.assignment.set_vdf('CQS')

        self.assignment.set_vdf('BPR')

    def test_set_classes(self):
        with self.assertRaises(ValueError):
            self.assignment.set_classes([1, 2])

        # The traffic assignment class is unprotected.
        # Should we protect it?
        # self.assigclass = TrafficClass(self.car_graph, self.matrix)
        # self.assigclass.graph = 1
        # with self.assertRaises(ValueError):
        #     self.assignment.set_classes(self.assigclass)

        self.assignment.set_classes(self.assigclass)
        # self.fail()

    def test_algorithms_available(self):
        algs = self.assignment.algorithms_available()
        real = ['all-or-nothing', 'msa', 'frank-wolfe', 'bfw', 'cfw', 'fw']

        diff = [x for x in real if x not in algs]
        diff2 = [x for x in algs if x not in real]

        if len(diff) + len(diff2) > 0:
            self.fail('list of algorithms raised is wrong')

    def test_set_cores(self):
        with self.assertRaises(Exception):
            self.assignment.set_cores(3)

        self.assignment.set_classes(self.assigclass)
        with self.assertRaises(ValueError):
            self.assignment.set_cores('q')

        self.assignment.set_cores(3)

    def test_set_algorithm(self):
        with self.assertRaises(AttributeError):
            self.assignment.set_algorithm('not an algo')

        self.assignment.set_classes(self.assigclass)

        with self.assertRaises(Exception):
            self.assignment.set_algorithm('msa')

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 10

        for algo in self.algorithms:
            for _ in range(10):
                algo = ''.join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])
                self.assignment.set_algorithm(algo)

    def test_set_vdf_parameters(self):
        with self.assertRaises(Exception):
            self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_vdf('bpr')
        self.assignment.set_classes(self.assigclass)
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

    def test_set_time_field(self):
        N = random.randint(1, 50)
        val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        self.assignment.set_time_field(val)
        self.assertEqual(self.assignment.time_field, val)

    def test_set_capacity_field(self):
        N = random.randint(1, 50)
        val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        self.assignment.set_capacity_field(val)
        self.assertEqual(self.assignment.capacity_field, val)

    def test_execute(self):

        self.assignment.set_classes(self.assigclass)
        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 10
        self.assignment.set_algorithm('msa')
        self.assignment.execute()
        msa10 = self.assignment.assignment.rgap

        self.assigclass.results.total_flows()
        correl = np.corrcoef(self.assigclass.results.total_link_loads, self.assigclass.graph.graph['volume'])[0, 1]
        self.assertLess(0.8, correl)

        self.assignment.max_iter = 30
        self.assignment.set_algorithm('msa')
        self.assignment.execute()
        msa25 = self.assignment.assignment.rgap

        self.assigclass.results.total_flows()
        correl = np.corrcoef(self.assigclass.results.total_link_loads, self.assigclass.graph.graph['volume'])[0, 1]
        self.assertLess(0.95, correl)

        self.assignment.set_algorithm('frank-wolfe')
        self.assignment.execute()
        fw25 = self.assignment.assignment.rgap

        self.assigclass.results.total_flows()
        correl = np.corrcoef(self.assigclass.results.total_link_loads, self.assigclass.graph.graph['volume'])[0, 1]
        self.assertLess(0.97, correl)

        self.assignment.set_algorithm('cfw')
        self.assignment.execute()
        cfw25 = self.assignment.assignment.rgap

        self.assigclass.results.total_flows()
        correl = np.corrcoef(self.assigclass.results.total_link_loads, self.assigclass.graph.graph['volume'])[0, 1]
        self.assertLess(0.98, correl)

        self.assignment.set_algorithm('bfw')
        self.assignment.execute()
        bfw25 = self.assignment.assignment.rgap

        self.assigclass.results.total_flows()
        correl = np.corrcoef(self.assigclass.results.total_link_loads, self.assigclass.graph.graph['volume'])[0, 1]
        self.assertLess(0.99, correl)

        self.assertLess(msa25, msa10)
        self.assertLess(fw25, msa25)
        self.assertLess(cfw25, fw25)
        self.assertLess(bfw25, cfw25)

    def test_info(self):
        iterations = random.randint(1, 10000)
        rgap = random.random() / 10000
        algo = choice(self.algorithms)

        self.assignment.set_classes(self.assigclass)
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
            algo = ''.join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])

        dct = self.assignment.info()
        if algo.lower() == 'fw':
            algo = 'frank-wolfe'
        self.assertEqual(dct['Algorithm'], algo.lower(), 'Algorithm not correct in info method')

        self.assertEqual(dct['Maximum iterations'], iterations, 'maximum iterations not correct in info method')
