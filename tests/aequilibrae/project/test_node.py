from unittest import TestCase
from sqlite3 import IntegrityError
import os
from shutil import copytree, rmtree
from random import randint, random
import uuid
from tempfile import gettempdir
from shapely.geometry import Point
import shapely.wkb
from aequilibrae.project import Project

from ...data import siouxfalls_project


class TestNode(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.curr.close()
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

        self.curr.execute("Select is_centroid, asBinary(geometry) from nodes where node_id=?;", [nd])
        flag, wkb = self.curr.fetchone()
        self.assertEqual(flag, 0, "Saving of is_centroid failed")

        geo = shapely.wkb.loads(wkb)
        self.assertEqual(geo.x, x, "Geometry X saved wrong")
        self.assertEqual(geo.y, y, "Geometry Y saved wrong")

        self.curr.execute("Select asBinary(geometry) from links where a_node=?;", [nd])
        wkb = self.curr.fetchone()[0]

        geo2 = shapely.wkb.loads(wkb)
        self.assertEqual(geo2.xy[0][0], x, "Saving node geometry broke underlying network")
        self.assertEqual(geo2.xy[1][0], y, "Saving node geometry broke underlying network")

    def test_data_fields(self):
        nodes = self.network.nodes

        node1 = nodes.get(randint(1, 24))
        node2 = nodes.get(randint(1, 24))

        self.assertEqual(node1.data_fields(), node2.data_fields(), "Different nodes have different data fields")

        fields = sorted(node1.data_fields())
        self.curr.execute("pragma table_info(nodes)")
        dt = self.curr.fetchall()

        actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

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

        self.curr.execute("Select asBinary(geometry) from nodes where node_id=?;", [num])
        wkb = self.curr.fetchone()[0]

        geo = shapely.wkb.loads(wkb)
        self.assertEqual(geo.x, x, "Renumbering failed")
        self.assertEqual(geo.y, y, "Renumbering failed")
