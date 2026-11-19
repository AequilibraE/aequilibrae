from math import pi
from random import randint

import pytest
import shapely
from shapely.geometry import LineString, Point

from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.transit_elements.link import Link

from .random_word import randomword
from .synthetic_gtfs import add_shape, add_stop, add_trip, base_tables, import_feed

# A shape and stops a little off it, as (lat, lon). Rounding in projecting the stops onto the shape puts the second
# stop of each pair at the same coordinates, A1 and A2 or C1 and C2, a few 1e-15 degrees further along it.
SHAPE = [(-29.95, -71.33), (-29.949, -71.322), (-29.95, -71.314)]
STOPS = {"A1": (-29.9499, -71.33), "A2": (-29.9499, -71.33), "B": (-29.9495, -71.326)}
STOPS |= {"C1": (-29.9499, -71.32), "C2": (-29.9499, -71.32), "D": (-29.95, -71.314)}


@pytest.fixture(scope="module")
def srid():
    return get_srid()


def test_build_object(srid):
    new_link = Link(srid)

    assert new_link.key == "####-1", "Pair not initiated properly"
    assert new_link.length == -1, "Length not initiated properly"

    assert new_link.srid == srid, "SRID was not assigned properly"

    fstop = randomword(randint(3, 15))
    tstop = randomword(randint(3, 15))

    new_link.from_stop = fstop
    assert new_link.key == fstop + "####-1", "Pair not computed properly"

    new_link.to_stop = tstop
    assert new_link.key == fstop + "##" + tstop + "##-1", "Pair not computed properly"

    geo = LineString([(0, 0), (3, 4)])

    new_link.geo = geo
    assert new_link.length == pytest.approx(5 * pi * 6371000 / 180), "Length not computed properly"


def test_save_to_database(srid, build_gtfs_project):
    geo = LineString([(0, 0), (3, 4)])
    new_link = Link(srid)

    with pytest.raises(AttributeError):
        new_link.save_to_database(build_gtfs_project.project.transit_connection)

    with build_gtfs_project.project.transit_connection as transit_conn:
        fstop, tstop = [row[0] for row in transit_conn.execute("SELECT stop_id FROM stops LIMIT 2")]
        pattern_id = transit_conn.execute("SELECT pattern_id FROM routes LIMIT 1").fetchone()[0]

        new_link.geo = geo
        new_link.transit_link = 10000001
        new_link.from_stop = fstop
        new_link.to_stop = tstop
        new_link.pattern_id = pattern_id
        new_link.seq = 4
        new_link.save_to_database(transit_conn, commit=False)

        from_stop, to_stop, dist = transit_conn.execute(
            "SELECT from_stop, to_stop, distance FROM route_links WHERE transit_link=?", [new_link.transit_link]
        ).fetchone()
        assert [str(from_stop), str(to_stop), round(dist * 180 / pi / 6371000)] == [
            fstop,
            tstop,
            geo.length,
        ], "Saving link to the database failed"


def test_build_geo_between_stops_at_the_same_coordinates(srid):
    shape = LineString([(lon, lat) for lat, lon in SHAPE])
    lat, lon = STOPS["A1"]
    stop = Point(lon, lat)

    link = Link(srid)
    assert link.build_geo(stop, stop, shape, stop, 0) == 0, "Position along the shape should not move"
    assert set(link.geo.coords) == {(lon, lat)}
    assert link.length == 0


def test_import_consecutive_stops_at_the_same_coordinates(empty_project, tmp_path):
    tables = base_tables()
    for stop, (lat, lon) in STOPS.items():
        add_stop(tables, stop, lat, lon)
    add_shape(tables, "S", SHAPE)
    add_trip(tables, "R", "T", [(stop, 25200 + 60 * i, 25200 + 60 * i) for i, stop in enumerate(STOPS)], shape_id="S")
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    sql = """SELECT a.stop, b.stop, l.distance, ST_AsBinary(l.geometry)
             FROM route_links l JOIN stops a ON a.stop_id = l.from_stop JOIN stops b ON b.stop_id = l.to_stop
             ORDER BY l.seq"""
    with empty_project.transit_connection as conn:
        links = {(fstop, tstop): (dist, shapely.from_wkb(wkb)) for fstop, tstop, dist, wkb in conn.execute(sql)}

    assert list(links) == [("A1", "A2"), ("A2", "B"), ("B", "C1"), ("C1", "C2"), ("C2", "D")]
    # The trip starts at A1, so the link between A1 and A2 is a line of length zero at the stops
    distance, geo = links[("A1", "A2")]
    assert distance == 0
    assert set(geo.coords) == {STOPS["A1"][::-1]}
    # The link to C1 ends on the shape, so the link between C1 and C2 joins it to the stops
    distance, geo = links[("C1", "C2")]
    assert distance > 0
    assert geo.coords[-1] == STOPS["C1"][::-1]
    assert LineString([(lon, lat) for lat, lon in SHAPE]).distance(Point(geo.coords[0])) < 1e-12
