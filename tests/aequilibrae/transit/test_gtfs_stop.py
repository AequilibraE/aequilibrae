from random import randint, choice, uniform
from shapely.geometry import LineString
import pytest
from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.get_srid import get_srid

from aequilibrae.transit.transit_elements import Stop
from tests.aequilibrae.transit.random_word import randomword


@pytest.fixture
def data():
    return {
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
    }


def test__populate(data):
    s = Stop(1, tuple(data.values()), list(data.keys()))
    xy = (s.geo.x, s.geo.y)
    assert xy == (data["stop_lon"], data["stop_lat"]), "Stop built geo wrongly"
    data["stop"] = data["stop_id"]
    data["stop_id"] = s.stop_id
    data["zone"] = data.pop("zone_id")

    for key, val in s.__dict__.items():
        if key in data:
            assert val == data[key], "Stop population with record failed"

    new_data = {key: val for key, val in data.items()}
    new_data[randomword(randint(1, 15))] = randomword(randint(1, 20))
    with pytest.raises(KeyError):
        s = Stop(1, tuple(new_data.values()), list(new_data.keys()))


def test_save_to_database(data, transit_conn):
    line = LineString([[-23.59, -46.64], [-23.43, -46.50]]).wkb
    tlink_id = randint(10000, 200000044)
    s = Stop(1, tuple(data.values()), list(data.keys()))
    s.link = link = randint(1, 30000)
    s.dir = direc = choice((0, 1))
    s.agency = randint(5, 100000)
    s.route_type = randint(0, 13)
    s.srid = get_srid()
    s.get_node_id()
    s.save_to_database(transit_conn, commit=True)

    sql_tl = """Insert into route_links ("transit_link", "pattern_id", "seq", "from_stop", "to_stop", "distance", "geometry")
                VALUES(?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326));"""
    transit_conn.execute(sql_tl, [tlink_id, randint(1, 1000000000), randint(1, 10), s.stop_id, s.stop_id + 1, 0, line])

    qry = transit_conn.execute(
        "Select agency_id, link, dir, description, street from stops where stop=?", [data["stop_id"]]
    ).fetchone()
    result = [x for x in qry]
    expected = [s.agency_id, link, direc, data["stop_desc"], data["stop_street"]]
    assert result == expected, "Saving Stop to the database failed"
