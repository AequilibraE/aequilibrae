from unittest import TestCase
import os
import pathlib
import sqlite3
import uuid
import string
import random
from random import choice
from tempfile import gettempdir
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

    def test_set_save_path_files(self):
        self.assignment.set_classes([self.assigclass])
        # make sure default is false
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, False)
        self.assignment.set_save_path_files(True)
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, True)

        # reset for most assignment tests
        self.assignment.set_save_path_files(False)
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, False)

    def test_set_path_file_format(self):
        self.assignment.set_classes([self.assigclass])
        with self.assertRaises(Exception):
            self.assignment.set_path_file_format("shiny_format")
        self.assignment.set_path_file_format("parquet")
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.write_feather, False)
        self.assignment.set_path_file_format("feather")
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.write_feather, True)

    def test_save_path_files(self):
        self.assignment.add_class(self.assigclass)
        self.assignment.set_save_path_files(True)

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 2
        self.assignment.set_algorithm("msa")
        self.assignment.execute()

        path_file_dir = pathlib.Path(self.project.project_base_path) / "path_files" / self.assignment.procedure_id
        self.assertTrue(path_file_dir.is_dir())

        # compare everything to reference files. Note that there is no graph simplification happening in SiouxFalls
        # and therefore we compare the files directly, otherwise a translation from the simplified ids to link_ids
        # would need to be performed.
        # Reference files were generated on 12/6/21, any changes to the test project will need to be applied to the
        # reference files. Also, the name given to the traffic class (see setUp above) has to be "car".
        class_id = f"c{self.assigclass.mode}_{self.assigclass.__id__}"
        reference_path_file_dir = pathlib.Path(siouxfalls_project) / "path_files"

        ref_node_correspondence = pd.read_feather(reference_path_file_dir / f"nodes_to_indeces_{class_id}.feather")
        node_correspondence = pd.read_feather(path_file_dir / f"nodes_to_indeces_{class_id}.feather")
        ref_node_correspondence.node_index = ref_node_correspondence.node_index.astype(node_correspondence.node_index.dtype)
        self.assertTrue(node_correspondence.equals(ref_node_correspondence))

        ref_correspondence = pd.read_feather(reference_path_file_dir / f"correspondence_{class_id}.feather")
        correspondence = pd.read_feather(path_file_dir / f"correspondence_{class_id}.feather")
        for col in correspondence.columns:
            ref_correspondence[col] = ref_correspondence[col].astype(correspondence[col].dtype)
        self.assertTrue(correspondence.equals(ref_correspondence))

        path_class_id = f"path_{class_id}"
        for i in range(1, self.assignment.max_iter + 1):
            class_dir = path_file_dir / f"iter{i}" / path_class_id
            ref_class_dir = reference_path_file_dir / f"iter{i}" / path_class_id
            for o in self.assigclass.matrix.index:
                o_ind = self.assigclass.graph.compact_nodes_to_indices[o]
                this_o_path_file = pd.read_feather(class_dir / f"o{o_ind}.feather")
                ref_this_o_path_file = pd.read_feather(ref_class_dir / f"o{o_ind}.feather")
                is_eq = this_o_path_file == ref_this_o_path_file
                self.assertTrue(is_eq.all().all())

                this_o_index_file = pd.read_feather(class_dir / f"o{o_ind}_indexdata.feather")
                ref_this_o_index_file = pd.read_feather(ref_class_dir / f"o{o_ind}_indexdata.feather")
                is_eq = this_o_index_file == ref_this_o_index_file
                self.assertTrue(is_eq.all().all())

        self.assignment.set_save_path_files(False)

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

        with self.assertRaises(ValueError):
            # We have no skimming setup
            self.assignment.save_skims("my_skims", "all")

        msa10 = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume.values)[0, 1]
        self.assertLess(0.8, correl)

        self.assignment.max_iter = 50
        self.assignment.set_algorithm("msa")
        self.assignment.execute()
        msa25 = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.98, correl)

        self.assignment.set_algorithm("frank-wolfe")
        self.assignment.execute()

        fw25 = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.99, correl)

        self.assignment.set_algorithm("cfw")
        self.assignment.execute()
        cfw25 = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.995, correl)

        # For the last algorithm, we set skimming
        self.car_graph.set_skimming(["free_flow_time", "distance"])
        assigclass = TrafficClass("car", self.car_graph, self.matrix)
        self.assignment.set_classes([assigclass])

        self.assignment.set_algorithm("bfw")
        self.assignment.execute()
        bfw25 = self.assignment.assignment.rgap

        correl = np.corrcoef(self.assigclass.results.total_link_loads, results.volume)[0, 1]
        self.assertLess(0.999, correl)

        self.assertLess(msa25, msa10)
        self.assertLess(fw25, msa25)
        self.assertLess(cfw25, fw25)
        self.assertLess(bfw25, cfw25)

        self.assignment.save_results("save_to_database")
        self.assignment.save_skims("my_skims", "all")

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
