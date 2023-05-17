from os.path import isfile, join
from aequilibrae.project.database_connection import database_connection


def test_new_gtfs_builder(create_gtfs_project, create_path):
    transit = create_gtfs_project.new_gtfs_builder(
        agency="Agency_1",
        day="2016-04-13",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"

    transit.save_to_disk()

    transit = create_gtfs_project.new_gtfs_builder(
        agency="Agency_2",
        day="2016-07-19",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"

    transit.save_to_disk()

    conn = database_connection("transit")

    assert conn.execute("SELECT MAX(DISTINCT(agency_id)) FROM agencies;").fetchone()[0] == 2


def test___create_transit_database(create_gtfs_project):
    assert isfile(join(create_gtfs_project.project_base_path, "public_transport.sqlite")) is True
