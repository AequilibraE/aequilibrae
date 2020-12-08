from warnings import warn
from uuid import uuid4
from random import randint
from shutil import copytree, rmtree
from os.path import join
from tempfile import gettempdir
from unittest import TestCase
from shapely.geometry import Point, MultiPolygon
import shapely.wkb
from aequilibrae import Project
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
            warn(f'Error: {e.args}')

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

        curr = self.proj.conn.cursor()
        curr.execute('Select area from Zones where zone_id=2')
        self.assertEqual(curr.fetchone()[0], area, "Zone didn't save area properly")

        geo = Point(0, 0).buffer(1)
        zn.geometry = geo
        zn.save()
        curr = self.proj.conn.cursor()
        curr.execute('Select asBinary(geometry) from Zones where zone_id=2')
        wkb = curr.fetchone()[0]
        self.assertEqual(shapely.wkb.loads(wkb), MultiPolygon([geo]), "Zone didn't save geometry properly")

        zn2 = zones.get(1)
        geo = MultiPolygon([Point(0, 0).buffer(1)])
        zn2.geometry = geo
        zn2.save()
        curr = self.proj.conn.cursor()
        curr.execute('Select asBinary(geometry) from Zones where zone_id=1')
        wkb = curr.fetchone()[0]
        self.assertEqual(shapely.wkb.loads(wkb), geo, "Zone didn't save geometry properly")
