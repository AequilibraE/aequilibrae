import os
from random import randint
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4

from numpy import array
from shapely.geometry import MultiLineString
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.functions.transit_connection import transit_connection

from aequilibrae.transit.transit_elements import Route
from aequilibrae.utils.create_example import create_example
from tests.aequilibrae.transit.random_word import randomword


class TestRoute(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = create_example(self.fldr, "nauru")
        self.prj.create_empty_transit()

        self.network = transit_connection(self.fldr)

        self.data = {
            "route_id": randomword(randint(0, 40)),
            "route_short_name": randomword(randint(0, 40)),
            "route_long_name": randomword(randint(0, 40)),
            "route_desc": randomword(randint(0, 40)),
            "route_type": randint(0, 13),
            "route_url": randomword(randint(0, 40)),
            "route_color": randomword(randint(0, 40)),
            "route_text_color": randomword(randint(0, 40)),
            "route_sort_order": randint(0, 2000),
            "agency_id": randint(0, 1000),
        }

    def tearDown(self) -> None:
        self.network.close()

    def test__populate(self):

        r = Route(1)
        r.populate(tuple(self.data.values()), list(self.data.keys()))
        self.data["route"] = self.data.pop("route_id")
        for key, val in r.__dict__.items():
            if key in self.data:
                self.assertEqual(val, self.data[key], "Route population with record failed")

        self.data[randomword(randint(1, 15))] = randomword(randint(1, 20))
        new_r = Route(1)
        with self.assertRaises(KeyError):
            new_r.populate(tuple(self.data.values()), list(self.data.keys()))

    def test_save_to_database(self):
        r = Route(1)
        r.srid = get_srid()
        r.populate(tuple(self.data.values()), list(self.data.keys()))
        r.shape = MultiLineString([array(((0.0, 0.0), (1.0, 2.0)))])
        r.save_to_database(self.network)
        curr = self.network.cursor()
        curr.execute(
            "Select agency_id, shortname, longname, description, route_type from routes where route=?",
            [self.data["route_id"]],
        )
        result = [x for x in curr.fetchone()]
        expected = [
            self.data["agency_id"],
            self.data["route_short_name"],
            self.data["route_long_name"],
            self.data["route_desc"],
            self.data["route_type"],
        ]
        self.assertEqual(result, expected, "Saving route to the database failed")
        del curr
