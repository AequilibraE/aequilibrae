from random import randint, choice
import pytest

from aequilibrae.transit.transit_elements import Trip
from tests.aequilibrae.transit.random_word import randomword


class TestTrip:
    @pytest.fixture
    def data(self):
        return {
            "route_id": randomword(randint(0, 40)),
            "service_id": randomword(randint(0, 40)),
            "trip_id": randomword(randint(0, 40)),
            "trip_headsign": randomword(randint(0, 5)),
            "trip_short_name": randomword(randint(0, 5)),
            "block_id": randomword(randint(0, 5)),
            "shape_id": randomword(randint(0, 5)),
            "direction_id": choice([0, 1]),
            "bikes_allowed": choice([0, 1]),
        }

    def test_populate(self, data):
        s = Trip()

        s._populate(tuple(data.values()), list(data.keys()))
        data["route"] = data.pop("route_id")
        data["trip"] = data.pop("trip_id")
        for key, val in s.__dict__.items():
            if key in data:
                assert val == data[key], "StopTime population with record failed"

        data[randomword(randint(1, 30))] = randomword(randint(1, 30))

        with pytest.raises(KeyError):
            s = Trip()
            s._populate(tuple(data.values()), list(data.keys()))

    def test_save_to_database(self, data, transit_conn):
        r = Trip()
        r._populate(tuple(data.values()), list(data.keys()))
        times = [r for r in range(randint(5, 15))]
        patid = randint(15, 2500000)

        r.arrivals = [r for r in times]
        r.departures = [r for r in times]
        r.pattern_id = patid
        r.source_time = [0] * len(times)
        r.save_to_database(transit_conn)

        result = transit_conn.execute("Select pattern_id from trips where trip_id=?", [r.trip_id]).fetchone()[0]
        assert result == patid, "Saving trip to trips failed"

        records, counter = transit_conn.execute(
            'Select count(*), max("seq") from trips_schedule where trip_id=?', [r.trip_id]
        ).fetchone()
        assert records == len(times), "Saving trip to trips_schedule failed"
        assert counter == max(times), "Saving trip to trips_schedule failed"
