from random import randint
import pytest

from numpy import array
from shapely.geometry import MultiLineString
from aequilibrae.transit.functions.get_srid import get_srid

from aequilibrae.transit.transit_elements import Route
from tests.aequilibrae.transit.random_word import randomword


class TestRoute:
    @pytest.fixture
    def data_dict(self):
        return {
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

    def test__populate(self, data_dict):
        data = data_dict

        r = Route(1)
        r.populate(tuple(data.values()), list(data.keys()))
        data["route"] = data.pop("route_id")
        for key, val in r.__dict__.items():
            if key in data:
                assert val == data[key], "Route population with record failed"

        data[randomword(randint(1, 15))] = randomword(randint(1, 20))
        new_r = Route(1)
        with pytest.raises(KeyError):
            new_r.populate(tuple(data.values()), list(data.keys()))

    def test_save_to_database(self, data_dict, transit_conn):
        data = data_dict

        r = Route(1)
        r.srid = get_srid()
        r.populate(tuple(data.values()), list(data.keys()))
        r.shape = MultiLineString([array(((0.0, 0.0), (1.0, 2.0)))])
        r.save_to_database(transit_conn)
        curr = transit_conn.cursor()
        curr.execute(
            "Select agency_id, shortname, longname, description, route_type from routes where route=?",
            [data["route_id"]],
        )
        result = [x for x in curr.fetchone()]
        expected = [
            data["agency_id"],
            data["route_short_name"],
            data["route_long_name"],
            data["route_desc"],
            data["route_type"],
        ]
        assert result == expected, "Saving route to the database failed"
        del curr
