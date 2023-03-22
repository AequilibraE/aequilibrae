from unittest import TestCase
import os
import tempfile
import numpy as np
from aequilibrae.paths import Graph
from os.path import join
from uuid import uuid4
from .parameters_test import centroids
from aequilibrae.project import Project
from ...data import siouxfalls_project
from aequilibrae.paths.results import PathResults

# Adds the folder with the data to the path and collects the paths to the files
# lib_path = os.path.abspath(os.path.join('..', '../tests'))
# sys.path.append(lib_path)
from ...data import path_test, test_graph, test_network
from shutil import copytree, rmtree


class TestGraph(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(tempfile.gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.project = Project()
        self.project.open(self.temp_proj_folder)
        self.project.network.build_graphs()
        self.graph = self.project.network.graphs["c"]

    def tearDown(self) -> None:
        self.project.close()

    def test_prepare_graph(self):
        graph = self.project.network.graphs["c"]
        graph.prepare_graph(np.arange(5) + 1)

    def test_set_graph(self):
        self.graph.set_graph(cost_field="distance")
        self.graph.set_blocked_centroid_flows(block_centroid_flows=True)
        self.assertEqual(self.graph.num_zones, 24, "Number of centroids not properly set")
        self.assertEqual(self.graph.num_links, 76, "Number of links not properly set")
        self.assertEqual(self.graph.num_nodes, 24, "Number of nodes not properly set - " + str(self.graph.num_nodes))

    def test_save_to_disk(self):
        self.graph.save_to_disk(join(path_test, "aequilibrae_test_graph.aeg"))
        self.graph_id = self.graph.__id__

    def test_load_from_disk(self):
        self.test_save_to_disk()
        reference_graph = Graph()
        reference_graph.load_from_disk(test_graph)

        new_graph = Graph()
        new_graph.load_from_disk(join(path_test, "aequilibrae_test_graph.aeg"))

    def test_available_skims(self):
        self.graph.prepare_graph(np.arange(5) + 1)
        avail = self.graph.available_skims()
        data_fields = [
            "distance",
            "name",
            "lanes",
            "capacity",
            "speed",
            "b",
            "free_flow_time",
            "power",
            "colum",
            "volume",
            "modes",
        ]
        for i in data_fields:
            if i not in avail:
                self.fail("Skim availability with problems")

    def test_exclude_links(self):
        # excludes a link before any setting or preparation
        self.graph.set_blocked_centroid_flows(False)

        self.graph.set_graph("distance")
        r1 = PathResults()
        r1.prepare(self.graph)
        r1.compute_path(20, 21)
        self.assertEqual(list(r1.path), [62])

        r1 = PathResults()
        self.graph.exclude_links([62])
        r1.prepare(self.graph)
        r1.compute_path(20, 21)
        self.assertEqual(list(r1.path), [63, 69])
