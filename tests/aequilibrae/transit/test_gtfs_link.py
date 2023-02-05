from math import pi
import sqlite3
from random import randint
import pytest

from shapely.geometry import LineString
from aequilibrae.transit.functions.get_srid import get_srid

from aequilibrae.transit.transit_elements import Link
from tests.aequilibrae.transit.random_word import randomword


class TestLink:
    @pytest.fixture
    def srid(self):
        return get_srid()

    def test_build_object(self, srid):
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

    def test_save_to_database(self, srid, transit_conn):
        route_type = randint(0, 13)
        fstop = randomword(randint(3, 15))
        tstop = randomword(randint(3, 15))
        geo = LineString([(0, 0), (3, 4)])

        new_link = Link(srid)

        with pytest.raises(AttributeError):
            new_link.save_to_database(transit_conn)

        new_link.geo = geo
        new_link.transit_link = 10000001
        new_link.type = route_type
        new_link.from_stop = fstop
        new_link.to_stop = tstop

        with pytest.raises(sqlite3.IntegrityError):
            new_link.save_to_database(transit_conn)

        new_link.pattern_id = 10001001000
        new_link.seq = 4

        new_link.save_to_database(transit_conn)

        from_stop, to_stop, dist = transit_conn.execute(
            "Select from_stop, to_stop, distance from route_links where from_stop=?", [fstop]
        ).fetchone()
        assert [from_stop, to_stop, round(dist * 180 / pi / 6371000)] == [
            fstop,
            tstop,
            geo.length,
        ], "Saving link to the database failed"
