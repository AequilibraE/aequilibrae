from unittest import TestCase
import os
import tempfile
import numpy as np
from aequilibrae.paths import TransitGraph
from os.path import join
from uuid import uuid4
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


class TestTransitGraphBuilder(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(tempfile.gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid4().hex)

        self.project = create_example(self.temp_proj_folder, "coquimbo")

        os.remove(os.path.join(self.temp_proj_folder, "public_transport.sqlite"))

        self.data = Transit(self.project)
        dest_path = join(self.temp_proj_folder, "gtfs_coquimbo.zip")
        self.transit = self.data.new_gtfs_builder(agency="LISANCO", file_path=dest_path)

        self.transit.load_date("2016-04-13")
        self.transit.save_to_disk()

    def tearDown(self) -> None:
        self.project.close()

    def test_create_line_gemoetry(self):
        self.project.network.build_graphs()
        for connector_method in ["overlapping_regions", "nearest_neighbour"]:
            for method in ["connector project match", "direct"]:
                with self.subTest(connector_method=connector_method, method=method):
                    graph = self.data.create_graph(
                        with_outer_stop_transfers=False,
                        with_walking_edges=False,
                        blocking_centroid_flows=False,
                        connector_method=connector_method,
                    )

                    self.assertNotIn("geometry", graph.edges.columns)

                    graph.create_line_geometry(method=method, graph="c")

                    self.assertIn("geometry", graph.edges.columns)
                    self.assertTrue(graph.edges.geometry.all())

    def test_connector_methods(self):
        connector_method = "nearest_neighbour"
        graph = self.data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method=connector_method,
        )

        nearest_neighbour_connector_count = len(graph.edges[graph.edges.link_type == "access_connector"])
        self.assertEqual(
            nearest_neighbour_connector_count, len(graph.edges[graph.edges.link_type == "egress_connector"])
        )
        self.assertEqual(
            nearest_neighbour_connector_count,
            len(graph.vertices[graph.vertices.node_type == "stop"]),
        )

        connector_method = "overlapping_regions"
        graph = self.data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method=connector_method,
        )

        self.assertLessEqual(
            nearest_neighbour_connector_count, len(graph.edges[graph.edges.link_type == "access_connector"])
        )
        self.assertEqual(
            len(graph.edges[graph.edges.link_type == "access_connector"]),
            len(graph.edges[graph.edges.link_type == "egress_connector"]),
        )

    def test_connector_method_exception(self):
        connector_method = "something not right"
        with self.assertRaises(ValueError):
            self.data.create_graph(
                with_outer_stop_transfers=False,
                with_walking_edges=False,
                blocking_centroid_flows=False,
                connector_method=connector_method,
            )

    def test_connector_method_without_missing(self):
        connector_method = "nearest_neighbour"
        graph = self.data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method=connector_method,
        )

        nearest_neighbour_connector_count = len(graph.edges[graph.edges.link_type == "access_connector"])
        self.assertEqual(
            nearest_neighbour_connector_count, len(graph.edges[graph.edges.link_type == "egress_connector"])
        )
        self.assertEqual(
            nearest_neighbour_connector_count,
            len(graph.vertices[graph.vertices.node_type == "stop"]),
        )

        connector_method = "overlapping_regions"
        graph = self.data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method=connector_method,
        )

        self.assertLessEqual(
            nearest_neighbour_connector_count, len(graph.edges[graph.edges.link_type == "access_connector"])
        )
        self.assertEqual(
            len(graph.edges[graph.edges.link_type == "access_connector"]),
            len(graph.edges[graph.edges.link_type == "egress_connector"]),
        )
