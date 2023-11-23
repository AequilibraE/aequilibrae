from copy import deepcopy
import os
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
import sys
from unittest import TestCase
from itertools import product

from aequilibrae.paths import Graph
from aequilibrae.paths.results import PathResults
from aequilibrae.utils.create_example import create_example
import numpy as np

# Adds the folder with the data to the path and collects the paths to the files
lib_path = os.path.abspath(os.path.join("..", "../tests"))
sys.path.append(lib_path)

origin = 5
dest = 13


class TestPathResultsDisconnected(TestCase):
    def setUp(self) -> None:
        self.project = create_example(join(gettempdir(), "test_path_disconnected" + uuid4().hex))

    def tearDown(self) -> None:
        self.project.close()

    def test_path_disconnected_delete_link(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                self.project.conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])
                self.project.conn.commit()

                self.project.network.build_graphs()
                self.g = self.project.network.graphs["c"]  # type: Graph
                self.g.set_graph("free_flow_time")
                self.g.set_blocked_centroid_flows(False)
                self.r = PathResults()
                self.r.prepare(self.g)
                self.r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
                self.assertEqual(None, self.r.path, "Failed to return None for disconnected")
                self.r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
                self.assertEqual(
                    len(self.r.path), 1, "Returned the wrong thing for existing path on disconnected network"
                )

    def test_path_disconnected_penalize_link_in_memory(self):
        for early_exit, a_star in product([True, False], repeat=2):
            with self.subTest(early_exit=early_exit, a_star=a_star):
                links = [2, 4, 5, 14]

                self.project.network.build_graphs()
                g = self.project.network.graphs["c"]  # type: Graph
                g.exclude_links(links)
                g.set_graph("free_flow_time")
                g.set_blocked_centroid_flows(False)
                r = PathResults()
                r.prepare(g)
                r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
                self.assertEqual(None, r.path, "Failed to return None for disconnected")
                r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
                self.assertEqual(len(r.path), 1, "Returned the wrong thing for existing path on disconnected network")
