from math import pi
from random import randint

import pytest
from shapely.geometry import LineString

from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.transit_elements import Link
from .random_word import randomword


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
