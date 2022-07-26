import os
import uuid
from copy import copy, deepcopy
from random import randint, random
from shutil import copytree, rmtree
from tempfile import gettempdir
from unittest import TestCase, TestLoader as _TestLoader

import shapely.wkb
from shapely.geometry import Point

from aequilibrae.project import Project
from ...data import siouxfalls_project

_TestLoader.sortTestMethodsUsing = None


class TestNodes(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.project.close()
        del self.curr
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f"Failed to remove at {e.args}")

    def test_get(self):
        nodes = self.network.nodes
        nd = randint(1, 24)
        node = nodes.get(nd)

        self.assertEqual(node.node_id, nd, "get node returned wrong object")

        # Make sure that if we renumber itg we would not get it again
        node.renumber(200)
        with self.assertRaises(ValueError):
            node = nodes.get(nd)

    def test_save(self):
        nodes = self.network.nodes
        chosen = [randint(1, 24) for _ in range(5)]
        while len(chosen) != len(set(chosen)):
            chosen = [randint(1, 24) for _ in range(5)]
        coords = []
        for nd in chosen:
            node = nodes.get(nd)
            node.is_centroid = 0
            x = node.geometry.x + random()
            y = node.geometry.y + random()
            coords.append([x, y])
            node.geometry = Point([x, y])

        nodes.save()
        for nd, coords in zip(chosen, coords):
            x, y = coords
            self.curr.execute("Select is_centroid, asBinary(geometry) from nodes where node_id=?;", [nd])
            flag, wkb = self.curr.fetchone()
            self.assertEqual(flag, 0, "Saving of is_centroid failed")

            geo = shapely.wkb.loads(wkb)
            self.assertEqual(geo.x, x, "Geometry X saved wrong")
            self.assertEqual(geo.y, y, "Geometry Y saved wrong")

    def test_fields(self):
        nodes = self.network.nodes
        f_editor = nodes.fields

        fields = sorted(f_editor.all_fields())
        self.curr.execute("pragma table_info(nodes)")
        dt = self.curr.fetchall()

        actual_fields = set([x[1] for x in dt if x[1] != "ogc_fid"])
        actual_fields = sorted(list(actual_fields))

        self.assertEqual(fields, actual_fields, "Table editor is weird for table nodes")

    def test_copy(self):
        nodes = self.network.nodes
        with self.assertRaises(Exception):
            _ = copy(nodes)
        with self.assertRaises(Exception):
            _ = deepcopy(nodes)

    def test_new_centroid(self):
        nodes = self.network.nodes

        with self.assertRaises(Exception):
            node = nodes.new_centroid(1)

        tot_prev_centr = self.network.count_centroids()
        tot_prev_nodes = self.network.count_nodes()
        node = nodes.new_centroid(100)
        self.assertEqual(1, node.is_centroid, "Creating new centroid returned wrong is_centroid value")
        node.geometry = Point(1, 1)
        node.save()

        self.assertEqual(tot_prev_centr + 1, self.network.count_centroids(), "Failed to add centroids")
        self.assertEqual(tot_prev_nodes + 1, self.network.count_nodes(), "Failed to add centroids")
