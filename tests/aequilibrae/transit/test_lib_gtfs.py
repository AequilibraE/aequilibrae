import os
import pytest
from uuid import uuid4

from aequilibrae.utils.create_example import create_example
from aequilibrae.project.database_connection import database_connection

from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder


@pytest.fixture
def create_path(tmp_path):
    return tmp_path / uuid4().hex


@pytest.fixture
def create_project(create_path):
    prj = create_example(create_path, "coquimbo")

    if os.path.isfile(os.path.join(create_path, "public_transport.sqlite")):
        os.remove(os.path.join(create_path, "public_transport.sqlite"))

    prj.create_empty_transit()
    yield prj
    prj.close()


@pytest.fixture
def network(create_project, create_path):
    return database_connection("transit", create_path)


@pytest.fixture
def gtfs_file():
    return os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/gtfs_coquimbo.zip")


@pytest.fixture
def system_builder(network, gtfs_file):

    yield GTFSRouteSystemBuilder(network=network, agency_identifier="LISERCO, LISANCO, LINCOSUR", file_path=gtfs_file)


def test_set_capacities(system_builder):
    system_builder.set_capacities({0: [150, 300], 3: [42, 56]})
    assert system_builder.gtfs_data.__dict__["__capacities__"] == {0: [150, 300], 3: [42, 56]}


def test_dates_available(system_builder):
    dates = system_builder.dates_available()
    assert type(dates) == list


def test_set_allow_map_match(system_builder):
    assert system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"] is False
    system_builder.set_allow_map_match(True)
    assert system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"] is True


def test_map_match_tuple_exception(system_builder):
    with pytest.raises(TypeError):
        system_builder.map_match(route_types=3)


def test_map_match_int_exception(system_builder):
    with pytest.raises(TypeError):
        system_builder.map_match(route_types=[3.5])


def test_map_match(network, system_builder):
    system_builder.load_date("2016-04-13")
    system_builder.set_allow_map_match(True)
    system_builder.map_match()
    system_builder.save_to_disk()

    assert network.execute("SELECT * FROM pattern_mapping;").fetchone()[0] > 1


def test_set_agency_identifier(system_builder):
    assert system_builder.gtfs_data.agency.agency != "CTA"
    system_builder.set_agency_identifier("CTA")
    assert system_builder.gtfs_data.agency.agency == "CTA"


def test_set_feed(gtfs_file, system_builder):
    system_builder.set_feed(gtfs_file)
    assert system_builder.gtfs_data.archive_dir == gtfs_file


def test_set_description(system_builder):
    system_builder.set_description("CTA2019 fixed by John Doe after strong coffee")
    assert system_builder.description == "CTA2019 fixed by John Doe after strong coffee"


def test_set_date(system_builder):
    system_builder.set_date("2016-04-13")
    assert system_builder.__target_date__ == "2016-04-13"


def test_load_date(system_builder):
    system_builder.load_date("2016-04-13")
    assert system_builder.gtfs_data.agency.service_date == "2016-04-13"
    assert "101387" in system_builder.select_routes.keys()


def test_load_date_srid_exception(system_builder):
    system_builder.srid = None
    with pytest.raises(ValueError):
        system_builder.load_date("2016-04-13")


def test_load_date_not_available_date_exception(system_builder):
    with pytest.raises(ValueError):
        system_builder.load_date("2020-06-01")


def test_set_do_raw_shapes(system_builder):
    system_builder.set_do_raw_shapes(True)
    assert system_builder.__do_raw_shapes__ is True


def test_create_raw_shapes(network, system_builder):
    system_builder.load_date("2016-04-13")
    system_builder.create_raw_shapes()

    all_tables = [x[0] for x in network.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]
    assert "raw_shapes" in all_tables


def test_save_to_disk(network, system_builder):

    system_builder.load_date("2016-04-13")
    system_builder.save_to_disk()

    assert len(network.execute("SELECT * FROM route_links").fetchall()) == 78
    assert len(network.execute("SELECT * FROM trips;").fetchall()) == 360
    assert len(network.execute("SELECT * FROM routes;").fetchall()) == 2
