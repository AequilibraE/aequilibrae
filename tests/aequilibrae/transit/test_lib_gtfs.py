import pytest

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder


@pytest.fixture(scope="function")
def route_system_builder(build_gtfs_project):
    gtfs_file = build_gtfs_project.project_base_path / "gtfs_coquimbo.zip"
    with database_connection("transit") as transit_conn:
        yield GTFSRouteSystemBuilder(
            network=transit_conn, agency_identifier="LISERCO, LISANCO, LINCOSUR", file_path=gtfs_file
        )


def test_set_capacities(route_system_builder):
    route_system_builder.set_capacities({0: [150, 300], 3: [42, 56]})
    assert route_system_builder.gtfs_data.__dict__["__capacities__"] == {0: [150, 300], 3: [42, 56]}


def test_set_pces(route_system_builder):
    route_system_builder.set_pces({1: 2.5, 3: 6.2})
    assert route_system_builder.gtfs_data.__dict__["__pces__"] == {1: 2.5, 3: 6.2}


def test_dates_available(route_system_builder):
    dates = route_system_builder.dates_available()
    assert isinstance(dates, list)


def test_set_allow_map_match(route_system_builder):
    assert route_system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"] is False
    route_system_builder.set_allow_map_match(True)
    assert route_system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"] is True


def test_map_match_tuple_exception(route_system_builder):
    with pytest.raises(TypeError):
        route_system_builder.map_match(route_types=3)


def test_map_match_int_exception(route_system_builder):
    with pytest.raises(TypeError):
        route_system_builder.map_match(route_types=[3.5])


def test_map_match(route_system_builder):
    route_system_builder.load_date("2016-04-13")
    route_system_builder.set_allow_map_match(True)
    route_system_builder.map_match()
    route_system_builder.save_to_disk()

    with database_connection("transit") as transit_conn:
        assert transit_conn.execute("SELECT * FROM pattern_mapping;").fetchone()[0] > 1


def test_set_agency_identifier(route_system_builder):
    assert route_system_builder.gtfs_data.agency.agency != "CTA"
    route_system_builder.set_agency_identifier("CTA")
    assert route_system_builder.gtfs_data.agency.agency == "CTA"


def test_set_feed(route_system_builder):
    assert route_system_builder.gtfs_data.archive_dir.stem == "gtfs_coquimbo"


def test_set_description(route_system_builder):
    route_system_builder.set_description("CTA2019 fixed by John Doe after strong coffee")
    assert route_system_builder.description == "CTA2019 fixed by John Doe after strong coffee"


def test_set_date(route_system_builder):
    route_system_builder.set_date("2016-04-13")
    assert route_system_builder.__target_date__ == "2016-04-13"


def test_load_date(route_system_builder):
    route_system_builder.load_date("2016-04-13")
    assert route_system_builder.gtfs_data.agency.service_date == "2016-04-13"
    assert "101387" in route_system_builder.select_routes.keys()


def test_load_date_srid_exception(route_system_builder):
    route_system_builder.srid = None
    with pytest.raises(ValueError):
        route_system_builder.load_date("2016-04-13")


def test_load_date_not_available_date_exception(route_system_builder):
    with pytest.raises(ValueError):
        route_system_builder.load_date("2020-06-01")


def test_save_to_disk(route_system_builder):
    route_system_builder.load_date("2016-04-13")
    route_system_builder.save_to_disk()

    with database_connection("transit") as transit_conn:
        assert len(transit_conn.execute("SELECT * FROM route_links").fetchall()) == 78
        assert len(transit_conn.execute("SELECT * FROM trips;").fetchall()) == 360
        assert len(transit_conn.execute("SELECT * FROM routes;").fetchall()) == 2
