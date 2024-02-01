import os
import uuid
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase
from warnings import warn

from shapely.geometry import box

from aequilibrae.project import Project
from ...data import siouxfalls_project


class TestNetwork(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.proj_path = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_path)
        self.siouxfalls = Project()
        self.siouxfalls.open(self.proj_path)
        self.proj_path2 = os.path.join(gettempdir(), uuid.uuid4().hex)

    def tearDown(self) -> None:
        self.siouxfalls.close()

    def test_create_from_osm(self):
        if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage":
            print("Skipped check to not load OSM servers")
            return

        self.siouxfalls.close()
        self.project = Project()
        self.project.new(self.proj_path2)
        self.project.network.create_from_osm(model_area=box(-112.185, 36.59, -112.179, 36.60))
        curr = self.project.conn.cursor()

        curr.execute("""select count(*) from links""")
        lks = curr.fetchone()[0]

        curr.execute("""select count(distinct osm_id) from links""")
        osmids = curr.fetchone()[0]

        if osmids == 0:
            warn("COULD NOT RETRIEVE DATA FROM OSM")
            return

        if osmids >= lks:
            self.fail("OSM links not broken down properly")

        curr.execute("""select count(*) from nodes""")
        nds = curr.fetchone()[0]

        if lks > nds:
            self.fail("We imported more links than nodes. Something wrong here")
        self.project.close()
        self.siouxfalls.open(self.proj_path)

    def test_count_centroids(self):
        items = self.siouxfalls.network.count_centroids()
        self.assertEqual(24, items, "Wrong number of centroids found")

        nodes = self.siouxfalls.network.nodes
        node = nodes.get(1)
        node.is_centroid = 0
        node.save()

        items = self.siouxfalls.network.count_centroids()
        self.assertEqual(23, items, "Wrong number of centroids found")

    def test_count_links(self):
        items = self.siouxfalls.network.count_links()
        self.assertEqual(76, items, "Wrong number of links found")

    def test_count_nodes(self):
        items = self.siouxfalls.network.count_nodes()
        self.assertEqual(24, items, "Wrong number of nodes found")
