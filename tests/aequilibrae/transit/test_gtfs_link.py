from math import pi
import os
from random import randint
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4

from shapely.geometry import LineString
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.get_srid import get_srid

from aequilibrae.transit.transit_elements import Link
from aequilibrae.utils.create_example import create_example
from tests.aequilibrae.transit.random_word import randomword


class TestLink(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = create_example(self.fldr, "nauru")
        self.prj.create_empty_transit()

        self.network = database_connection(table_type="transit")
        self.srid = get_srid()

    def tearDown(self) -> None:
        self.network.close()

    def test_build_object(self):
        new_link = Link(self.srid)

        self.assertEqual(new_link.key, "####-1", "Pair not initiated properly")
        self.assertEqual(new_link.length, -1, "Length not initiated properly")

        self.assertEqual(new_link.srid, self.srid, "SRID was not assigned properly")

        self.fstop = randomword(randint(3, 15))
        self.tstop = randomword(randint(3, 15))

        new_link.from_stop = self.fstop
        self.assertEqual(new_link.key, self.fstop + "####-1", "Pair not computed properly")

        new_link.to_stop = self.tstop
        self.assertEqual(new_link.key, self.fstop + "##" + self.tstop + "##-1", "Pair not computed properly")

        geo = LineString([(0, 0), (3, 4)])

        new_link.geo = geo
        self.assertEqual(round(new_link.length * 180 / pi / 6371000), 5, "Length not computed properly") 

    def test_save_to_database(self):
        route_type = randint(0, 13)
        fstop = randomword(randint(3, 15))
        tstop = randomword(randint(3, 15))
        geo = LineString([(0, 0), (3, 4)])

        new_link = Link(self.srid)
        new_link.type = route_type
        new_link.from_stop = fstop
        new_link.to_stop = tstop
        new_link.geo = geo

        new_link.save_to_database(self.network)

        from_stop, to_stop, dist = self.network.execute(
            'Select from_stop, to_stop, "length" from route_links where from_stop=?', [fstop]
        ).fetchone()
        self.assertEqual([from_stop, to_stop, round(dist * 180 / pi / 6371000)], [fstop, tstop, geo.length], "Saving link to the database failed")
