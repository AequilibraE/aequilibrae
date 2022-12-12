import os
from random import randint, choice
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.functions.transit_connection import transit_connection

from aequilibrae.transit.transit_elements import Trip
from aequilibrae.utils.create_example import create_example
from tests.aequilibrae.transit.random_word import randomword


class TestTrip(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = create_example(self.fldr, "nauru")
        self.prj.create_empty_transit()

        self.network = transit_connection(self.fldr)
        self.srid = get_srid()

        self.data = {
            "route_id": randomword(randint(0, 40)),
            "service_id": randomword(randint(0, 40)),
            "trip_id": randomword(randint(0, 40)),
            "trip_headsign": randomword(randint(0, 5)),
            "trip_short_name": randomword(randint(0, 5)),
            "block_id": randomword(randint(0, 5)),
            "shape_id": randomword(randint(0, 5)),
            "direction_id": choice([0, 1]),
            "wheelchair_accessible": choice([0, 1]),
            "bikes_allowed": choice([0, 1]),
        }

    def tearDown(self) -> None:
        self.network.close()

    def test_populate(self):
        s = Trip()

        s._populate(tuple(self.data.values()), list(self.data.keys()))
        self.data["route"] = self.data.pop("route_id")
        self.data["trip"] = self.data.pop("trip_id")
        for key, val in s.__dict__.items():
            if key in self.data:
                self.assertEqual(val, self.data[key], "StopTime population with record failed")

        self.data[randomword(randint(1, 30))] = randomword(randint(1, 30))

        with self.assertRaises(KeyError):
            s = Trip()
            s._populate(tuple(self.data.values()), list(self.data.keys()))

    def test_save_to_database(self):
        r = Trip()
        r._populate(tuple(self.data.values()), list(self.data.keys()))
        times = [r for r in range(randint(5, 15))]
        patid = randint(15, 2500000)

        r.arrivals = [r for r in times]
        r.departures = [r for r in times]
        r.pattern_id = patid
        r.source_time = [0] * len(times)
        r.save_to_database(self.network)

        result = self.network.execute("Select pattern_id from trips where trip_id=?", [r.trip_id]).fetchone()[0]
        self.assertEqual(result, patid, "Saving trip to trips failed")

        records, counter = self.network.execute('Select count(*), max("seq") from trips_schedule where trip_id=?', [r.trip_id]).fetchone()
        self.assertEqual(records, len(times), "Saving trip to trips_schedule failed")
        self.assertEqual(counter, max(times), "Saving trip to trips_schedule failed")
