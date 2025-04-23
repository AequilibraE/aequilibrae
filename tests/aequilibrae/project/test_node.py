import os
import uuid
from random import randint, random
from shutil import copytree, rmtree
from sqlite3 import IntegrityError
from tempfile import gettempdir
from unittest import TestCase

import shapely.wkb
from shapely.geometry import Point

from aequilibrae.project import Project
from aequilibrae.utils.db_utils import read_and_close
from ...data import siouxfalls_project


class TestNode(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network

    def tearDown(self) -> None:
        self.project.close()
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f"Failed to remove at {e.args}")

    def test_save_and_assignment(self):
        nodes = self.network.nodes
        nd = randint(1, 24)
        node = nodes.get(nd)

        with self.assertRaises(AttributeError):
            node.modes = "abc"

        with self.assertRaises(AttributeError):
            node.link_types = "default"

        with self.assertRaises(AttributeError):
            node.node_id = 2

        with self.assertRaises(ValueError):
            node.is_centroid = 2

        node.is_centroid = 0
        self.assertEqual(0, node.is_centroid, "Assignment of is_centroid did not work")

        x = node.geometry.x + random()
        y = node.geometry.y + random()

        node.geometry = Point([x, y])

        node.save()

        with read_and_close(self.project.path_to_file, spatial=True) as conn:
            sql = f"Select is_centroid, asBinary(geometry) from nodes where node_id={nd};"
            flag, wkb = conn.execute(sql).fetchone()
            self.assertEqual(flag, 0, "Saving of is_centroid failed")

            geo = shapely.wkb.loads(wkb)
            self.assertEqual(geo.x, x, "Geometry X saved wrong")
            self.assertEqual(geo.y, y, "Geometry Y saved wrong")

            sql = f"Select asBinary(geometry) from links where a_node={nd};"
            wkb = conn.execute(sql).fetchone()[0]

            geo2 = shapely.wkb.loads(wkb)
            self.assertEqual(geo2.xy[0][0], x, "Saving node geometry broke underlying network")
            self.assertEqual(geo2.xy[1][0], y, "Saving node geometry broke underlying network")

    def test_data_fields(self):
        nodes = self.network.nodes

        node1 = nodes.get(randint(1, 24))
        node2 = nodes.get(randint(1, 24))

        self.assertEqual(node1.data_fields(), node2.data_fields(), "Different nodes have different data fields")

        with read_and_close(self.project.path_to_file) as conn:
            dt = conn.execute("pragma table_info(nodes)").fetchall()
            actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

        fields = sorted(node1.data_fields())
        self.assertEqual(fields, actual_fields, "Node has unexpected set of fields")

    def test_renumber(self):
        nodes = self.network.nodes

        node = nodes.get(randint(2, 24))

        x = node.geometry.x
        y = node.geometry.y

        with self.assertRaises(IntegrityError):
            node.renumber(1)

        num = randint(25, 2000)
        node.renumber(num)

        with read_and_close(self.project.path_to_file, spatial=True) as conn:
            sql = f"Select asBinary(geometry) from nodes where node_id={num};"
            wkb = conn.execute(sql).fetchone()[0]

        geo = shapely.wkb.loads(wkb)
        self.assertEqual(geo.x, x, "Renumbering failed")
        self.assertEqual(geo.y, y, "Renumbering failed")
