from unittest import TestCase
import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
from aequilibrae.paths import Graph
from os.path import join, dirname
from uuid import uuid4
from .parameters_test import centroids
from aequilibrae.project import Project
from ...data import siouxfalls_project
from aequilibrae.paths.results import PathResults
from aequilibrae.utils.create_example import create_example
from aequilibrae.transit import Transit


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
        self.graph_id = self.graph._id

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


class TestTransitGraph(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(tempfile.gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid4().hex)

        self.project = create_example(self.temp_proj_folder, "coquimbo")

        self.data = Transit(self.project)

        self.graph = self.data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method="nearest_neighbour",
        )

        self.transit_graph = self.graph.to_transit_graph()

    def tearDown(self) -> None:
        self.project.close()

    def test_transit_graph_config(self):
        self.assertEqual(self.graph.config, self.transit_graph._config)

    def test_transit_graph_od_node_mapping(self):
        pd.testing.assert_frame_equal(self.graph.od_node_mapping, self.transit_graph.od_node_mapping)


class TestGraphCompression(TestCase):
    def setUp(self) -> None:
        proj_path = os.path.join(tempfile.gettempdir(), "test_graph_compression" + uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "KaiTang.zip")).extractall(proj_path)

        # proj_path = "/home/jake/Software/aequilibrae_performance_tests/models/kaitang"
        self.link_df = pd.read_csv(os.path.join(proj_path, "links_modified.csv"))
        self.node_df = pd.read_csv(os.path.join(proj_path, "nodes_modified.csv"))
        centroids_array = np.array([7, 8, 11])

        self.graph = Graph()
        self.graph.network = self.link_df
        self.graph.mode = "a"
        self.graph.prepare_graph(centroids_array)
        self.graph.set_blocked_centroid_flows(False)
        self.graph.set_graph("fft")

    def test_compressed_graph(self):
        # Check the compressed links, links 4 and 5 should be collapsed into 2 links from 3 - 10 and 10 - 3.
        compressed_links = self.graph.graph[
            self.graph.graph.__compressed_id__.duplicated(keep=False)
            & (self.graph.graph.__compressed_id__ != self.graph.compact_graph.id.max() + 1)
        ]

        self.assertListEqual(compressed_links.link_id.unique().tolist(), [4, 5])

        # Confirm these compacted links map back up to a contraction between the correct nodes
        self.assertListEqual(
            self.graph.compact_all_nodes[
                self.graph.compact_graph[self.graph.compact_graph.id.isin(compressed_links.__compressed_id__.unique())][
                    ["a_node", "b_node"]
                ].values
            ].tolist(),
            [[3, 10], [10, 3]],
        )

    def test_dead_end_removal(self):
        # The dead end remove should be able to remove links [30, 38]. In it's current state it is not able to remove
        # link 40 as it's a single direction link with no outgoing edges so its not possible to find the incoming edges
        # (in general) without a transposed graph representation.
        self.assertSetEqual(
            set(self.graph.dead_end_links),
            set(self.graph.graph[self.graph.graph.dead_end == 1].link_id) - {40},
            "Dead end removal removed incorrect links",
        )
