from unittest import TestCase
from tempfile import gettempdir
import os
import uuid
from shutil import copytree
from aequilibrae.project import Project
from aequilibrae.project.project_creation import remove_triggers, add_triggers
from ...data import siouxfalls_project
from shapely.geometry import LineString, Point


class TestNetworkTriggers(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.proj_path = os.path.join(gettempdir(), f"aeq_{uuid.uuid4().hex}")
        copytree(siouxfalls_project, self.proj_path)
        self.siouxfalls = Project()
        self.siouxfalls.open(self.proj_path)
        remove_triggers(self.siouxfalls.conn, self.siouxfalls.logger, db_type="network")
        add_triggers(self.siouxfalls.conn, self.siouxfalls.logger, db_type="network")

    def tearDown(self) -> None:
        self.siouxfalls.close()

    def test_delete_links_delete_nodes(self):
        items = self.siouxfalls.network.count_nodes()
        self.assertEqual(24, items, "Wrong number of nodes found")
        links = self.siouxfalls.network.links
        nodes = self.siouxfalls.network.nodes

        node = nodes.get(1)
        node.is_centroid = 0
        node.save()

        # We have completely disconnected 2 nodes (1 and 2)
        for i in [1, 2, 3, 4, 5, 14]:
            link = links.get(i)
            link.delete()
        # Since node 1 is no longer a centroid, we should have only 23 nodes in the network
        items = self.siouxfalls.network.count_nodes()
        self.assertEqual(23, items, "Wrong number of nodes found")

    def test_add_regular_link(self):
        # Add a regular link to see if it fails when creating it
        # It happened at some point
        curr = self.siouxfalls.conn.cursor()
        data = [123456, "c", "default", LineString([Point(0, 0), Point(1, 1)]).wkb]

        sql = "insert into links (link_id, modes, link_type, geometry) Values(?,?,?,GeomFromWKB(?, 4326));"
        curr.execute(sql, data)
        self.siouxfalls.conn.commit()

    def test_add_regular_node_change_centroid_id(self):
        # Add a regular link to see if it fails when creating it
        # It happened at some point
        curr = self.siouxfalls.conn.cursor()
        network = self.siouxfalls.network
        nodes = network.count_nodes()

        data = [987654, 1, Point(0, 0).wkb]

        sql = "insert into nodes (node_id, is_centroid, geometry) Values(?,?,GeomFromWKB(?, 4326));"
        curr.execute(sql, data)
        self.siouxfalls.conn.commit()
        self.assertEqual(nodes + 1, network.count_nodes(), "Failed to insert node")

        curr.execute("Update nodes set is_centroid=0 where node_id=?", data[:1])
        self.siouxfalls.conn.commit()
        self.assertEqual(nodes, network.count_nodes(), "Failed to delete node when changing centroid flag")
