import os
from random import randint, choice, uniform
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.get_srid import get_srid

from aequilibrae.transit.transit_elements import Stop

from tests.aequilibrae.transit.random_word import randomword


class TestStop(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

        self.network = database_connection(table_type="transit")
        self.srid = get_srid()

        self.data = {
            "stop_id": randint(0, 400000000),
            "stop_code": randomword(randint(0, 40)),
            "stop_name": randomword(randint(0, 40)),
            "stop_desc": randomword(randint(0, 40)),
            "stop_lat": uniform(0, 30000),
            "stop_lon": uniform(0, 30000),
            "stop_street": randomword(randint(0, 40)),
            "zone_id": randomword(randint(0, 40)),
            "stop_url": randomword(randint(0, 40)),
            "location_type": choice((0, 1)),
            "parent_station": randomword(randint(0, 40)),
            "stop_timezone": randint(0, 2000),
            "wheelchair_boarding": choice((0, 1)),
        }

    def tearDown(self) -> None:
        self.prj.close()

    def test__populate(self):
        s = Stop(1)
        s.populate(tuple(self.data.values()), list(self.data.keys()))
        xy = (s.geo.x, s.geo.y)
        self.assertEqual(xy, (self.data["stop_lon"], self.data["stop_lat"]), "Stop built geo wrongly")
        self.data["stop"] = self.data["stop_id"]
        self.data["stop_id"] = s.stop_id
        self.data["zone"] = self.data.pop("zone_id")

        for key, val in s.__dict__.items():
            if key in self.data:
                self.assertEqual(val, self.data[key], "Stop population with record failed")

        new_data = {key: val for key, val in self.data.items()}
        new_data[randomword(randint(1, 15))] = randomword(randint(1, 20))
        s = Stop(1)
        with self.assertRaises(KeyError):
            s.populate(tuple(new_data.values()), list(new_data.keys()))

    def test_save_to_database(self):
        tlink_id = randint(10000, 200000044)
        s = Stop(1)
        s.populate(tuple(self.data.values()), list(self.data.keys()))
        s.link = link = randint(1, 30000)
        s.dir = direc = choice((0, 1))
        s.agency = randint(5, 100000)
        s.route_type = randint(0, 13)
        s.srid = self.srid
        s.get_node_id()

        sql_tl = """Insert into route_links ("transit_link", "pattern_id", "from_stop", "to_stop", "length", "type")
                  VALUES(?, ?, ?, ?, 5, -1)"""

        self.network.execute(sql_tl, [tlink_id, randint(1, 1000000000), s.stop_id, s.stop_id + 1])
        s.save_to_database(self.network, commit=True)

        qry = self.network.execute("Select agency_id, link, dir, description, street from stops where stop=?", [self.data["stop_id"]]).fetchone()
        result = [x for x in qry]
        expected = [s.agency_id, link, direc, self.data["stop_desc"], self.data["stop_street"]]
        self.assertEqual(result, expected, "Saving Stop to the database failed")
