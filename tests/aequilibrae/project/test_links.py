import os
import uuid
from copy import copy, deepcopy
from random import randint
from shutil import copytree, rmtree
from tempfile import gettempdir
from unittest import TestCase

from aequilibrae.project import Project
from ...data import siouxfalls_project


class TestLinks(TestCase):
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
        links = self.network.links
        with self.assertRaises(ValueError):
            _ = links.get(123456)

        link = links.get(1)
        self.assertEqual(link.capacity_ab, 25900.20064, "Did not populate link correctly")

    def test_new(self):
        links = self.network.links
        new_link = links.new()

        self.curr.execute("Select max(link_id) + 1 from Links")
        id = self.curr.fetchone()[0]
        self.assertEqual(new_link.link_id, id, "Did not populate new link ID properly")
        self.assertEqual(new_link.geometry, None, "Did not populate new geometry properly")

    def test_copy_link(self):
        links = self.network.links

        with self.assertRaises(ValueError):
            _ = links.copy_link(11111)

        new_link = links.copy_link(11)

        old_link = links.get(11)

        self.assertEqual(new_link.geometry, old_link.geometry)
        self.assertEqual(new_link.a_node, old_link.a_node)
        self.assertEqual(new_link.b_node, old_link.b_node)
        self.assertEqual(new_link.direction, old_link.direction)
        self.assertEqual(new_link.distance, old_link.distance)
        self.assertEqual(new_link.modes, old_link.modes)
        self.assertEqual(new_link.link_type, old_link.link_type)
        new_link.save()

    def test_delete(self):
        links = self.network.links

        _ = links.get(10)

        self.curr.execute("Select count(*) from Links")
        tot = self.curr.fetchone()[0]
        links.delete(10)
        links.delete(11)
        self.curr.execute("Select count(*) from Links")
        tot2 = self.curr.fetchone()[0]
        self.assertEqual(tot, tot2 + 2, "Did not delete the link properly")

        with self.assertRaises(ValueError):
            links.delete(123456)

        with self.assertRaises(ValueError):
            _ = links.get(10)

    def test_fields(self):
        links = self.network.links
        f_editor = links.fields

        fields = sorted(f_editor.all_fields())
        self.curr.execute("pragma table_info(links)")
        dt = self.curr.fetchall()

        actual_fields = set([x[1].replace("_ab", "").replace("_ba", "") for x in dt if x[1] != "ogc_fid"])
        actual_fields = sorted(list(actual_fields))

        self.assertEqual(fields, actual_fields, "Table editor is weird for table links")

    def test_refresh(self):
        links = self.network.links

        link1 = links.get(1)
        val = randint(1, 99999999)
        original_value = link1.capacity_ba

        link1.capacity_ba = val
        link1_again = links.get(1)
        self.assertEqual(link1_again.capacity_ba, val, "Did not preserve correctly")

        links.refresh()
        link1 = links.get(1)
        self.assertEqual(link1.capacity_ba, original_value, "Did not reset correctly")

    def test_copy(self):
        nodes = self.network.nodes
        with self.assertRaises(Exception):
            _ = copy(nodes)
        with self.assertRaises(Exception):
            _ = deepcopy(nodes)
