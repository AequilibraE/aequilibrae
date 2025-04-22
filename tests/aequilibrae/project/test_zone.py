import sqlite3
from math import sqrt
from os.path import join
from random import randint
from shutil import copytree, rmtree
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from warnings import warn

import shapely.wkb
from shapely.geometry import Point, MultiPolygon, LineString, MultiLineString

from aequilibrae import Project
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.db_utils import read_and_close
from ...data import siouxfalls_project


class TestZone(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = join(gettempdir(), uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)

    def tearDown(self) -> None:
        self.proj.close()
        try:
            rmtree(self.temp_proj_folder)
        except Exception as e:
            warn(f"Error: {e.args}")

    def test_delete(self):
        zones = self.proj.zoning
        zone_downtown = zones.get(3)
        zone_downtown.delete()

        with self.assertRaises(ValueError):
            _ = zones.get(3)

    def test_save(self):
        zones = self.proj.zoning
        zn = zones.get(2)
        area = randint(0, 9999999999)
        zn.area = area
        zn.save()

        with read_and_close(self.proj.path_to_file, spatial=True) as conn:
            cnt = conn.execute("Select area from Zones where zone_id=2").fetchone()[0]
            self.assertEqual(cnt, area, "Zone didn't save area properly")

            geo = Point(0, 0).buffer(1)
            zn.geometry = geo
            zn.save()
            wkb = conn.execute("Select asBinary(geometry) from Zones where zone_id=2").fetchone()[0]
            self.assertEqual(shapely.wkb.loads(wkb), MultiPolygon([geo]), "Zone didn't save geometry properly")

            zn2 = zones.get(1)
            geo = MultiPolygon([Point(0, 0).buffer(1)])
            zn2.geometry = geo
            zn2.save()
            wkb = conn.execute("Select asBinary(geometry) from Zones where zone_id=1").fetchone()[0]
            self.assertEqual(shapely.wkb.loads(wkb), geo, "Zone didn't save geometry properly")

    def __change_project(self):
        self.proj.close()
        self.proj = Project()
        self.proj = create_example(join(gettempdir(), uuid4().hex), "nauru")
        zones = 5
        network = self.proj.network
        nodes = network.nodes

        geo = network.convex_hull()

        zone_area = geo.area / zones
        zone_side = sqrt(2 * sqrt(3) * zone_area / 9)

        extent = network.extent()
        b = extent.bounds

        with read_and_close(self.proj.path_to_file, spatial=True) as conn:
            sql = "select st_asbinary(HexagonalGrid(GeomFromWKB(?), ?, 0, GeomFromWKB(?)))"
            grid = conn.execute(sql, [extent.wkb, zone_side, Point(b[2], b[3]).wkb]).fetchone()[0]

        grid = shapely.wkb.loads(grid)

        grid = [p for p in grid.geoms if p.intersects(geo)]

        zoning = self.proj.zoning
        for i, zone_geo in enumerate(grid):
            zone = zoning.new(i + 1)
            zone.geometry = zone_geo
            zone.save()

            node = nodes.get(i + 1)
            node.renumber(i + 10001)

    def test_add_centroid(self):
        self.__change_project()
        zones = self.proj.zoning
        nodes = self.proj.network.nodes
        network = self.proj.network
        zone1 = zones.get(1)
        tot = network.count_centroids()
        zone1.add_centroid(None)
        self.assertEqual(tot + 1, network.count_centroids(), "Added less than it should've")

        tot = network.count_centroids()
        zone1.add_centroid(None)
        zone1.add_centroid(Point(0, 0))
        self.assertEqual(tot, network.count_centroids(), "Added more than should've")
        node1 = nodes.get(1)
        self.assertEqual(node1.geometry, zone1.geometry.centroid)

        zone2 = zones.get(2)
        zone2.add_centroid(Point(0, 0))

        node2 = nodes.get(2)
        self.assertEqual(node2.geometry, Point(0, 0))

        # Tests ne behaviour to deal with centroids that would fall exactly on top of existing nodes
        point_that_should = zone1.geometry.centroid
        self.__change_project()
        zones = self.proj.zoning
        network = self.proj.network
        nd = network.nodes.get(1000)
        nd.geometry = point_that_should
        nd.save()

        zone1 = zones.get(1)
        with self.assertRaises(sqlite3.IntegrityError):
            zone1.add_centroid(None, robust=False)
        zone1.add_centroid(None, robust=True)

    def test_connect_mode(self):
        self.__change_project()
        zones = self.proj.zoning

        zone1 = zones.get(1)
        zone1.add_centroid(None)

        with read_and_close(self.proj.path_to_file) as conn:
            zone1.connect_mode("c")

            cnt = conn.execute("Select count(*) from links where a_node=?", [1]).fetchone()[0]
            self.assertIsNot(0, cnt, "failed to add connectors")

            zone1.connect_mode("t")
            sql = """Select count(*) from links where a_node=? and instr(modes,'t')>0"""
            cnt = conn.execute(sql, [1]).fetchone()[0]
            self.assertIsNot(0, cnt, "failed to add connectors for mode t")

            # Cannot connect a centroid that does not exist
            zone2 = zones.get(2)
            zone2.connect_mode("c", conn=conn)

    def test_disconnect_mode(self):
        self.__change_project()
        zones = self.proj.zoning
        zone1 = zones.get(1)
        zone1.add_centroid(None)

        with read_and_close(self.proj.path_to_file) as conn:
            zone1.connect_mode("c")
            zone1.connect_mode("w")
            tot = conn.execute("""select COUNT(*) from links where a_node=1""").fetchone()[0]
            conn.execute("""Update links set modes = modes || 'w' where instr(modes,'w')=0""")

        zone1.disconnect_mode("w")

        with read_and_close(self.proj.path_to_file, spatial=True) as conn:
            cnt = conn.execute("""select COUNT(*) from links where a_node=1""").fetchone()[0]
            self.assertIsNot(tot, cnt, "failed to delete links")

            cnt = conn.execute("""Select count(*) from links where a_node=1 and instr(modes,'w')>0""").fetchone()[0]
            self.assertEqual(cnt, 0, "Failed to remove mode from all connectors")

    def test_get_closest_zone(self):
        pt_in = Point(-96.7716, 43.6069)
        pt_out = Point(-96.7754, 43.5664)
        self.assertEqual(self.proj.zoning.get_closest_zone(pt_in), 1)
        self.assertEqual(self.proj.zoning.get_closest_zone(pt_out), 3)

        line_in = LineString([(-96.7209, 43.6132), (-96.7033, 43.61316)])
        line_out = LineString([(-96.7473, 43.6046), (-96.7341, 43.6046)])
        self.assertEqual(self.proj.zoning.get_closest_zone(line_in), 2)
        self.assertEqual(self.proj.zoning.get_closest_zone(line_out), 2)

        multi_line_in = MultiLineString(
            [((-96.7589, 43.5692), (-96.7531, 43.5807)), ((-96.7531, 43.5807), (-96.7504, 43.5704))]
        )
        multi_line_out = MultiLineString(
            [((-96.7716, 43.5769), (-96.7683, 43.5801)), ((-96.7683, 43.5801), (-96.7574, 43.5784))]
        )
        self.assertEqual(self.proj.zoning.get_closest_zone(multi_line_in), 3)
        self.assertEqual(self.proj.zoning.get_closest_zone(multi_line_out), 3)
