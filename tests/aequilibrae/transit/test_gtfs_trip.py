from copy import copy
from random import choice, randint

import pytest

from aequilibrae.transit.transit_elements import Trip

from .random_word import randomword


@pytest.fixture
def data():
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


def test_populate(data):
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


def test_save_to_database(build_gtfs_project, data):
    r = Trip()
    r._populate(tuple(data.values()), list(data.keys()))
    times = list(range(randint(5, 15)))
    patid = 10001001000  # Pattern ID must exist

    r.arrivals = copy(times)
    r.departures = copy(times)
    r.pattern_id = patid
    r.source_time = [0] * len(times)

    with build_gtfs_project.transit_connection as transit_conn:
        r.save_to_database(transit_conn, commit=False)
        result = transit_conn.execute("Select pattern_id from trips where trip_id=?", [r.trip_id]).fetchone()[0]

        records, counter = transit_conn.execute(
            'Select count(*), max("seq") from trips_schedule where trip_id=?', [r.trip_id]
        ).fetchone()

    assert result == patid, "Saving trip to trips failed"
    assert records == len(times), "Saving trip to trips_schedule failed"
    assert counter == max(times), "Saving trip to trips_schedule failed"
