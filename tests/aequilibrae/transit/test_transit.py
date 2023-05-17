from os.path import isfile, join

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.constants import Constants


def test_new_gtfs_builder(create_gtfs_project, create_path):
    c = Constants()
    c.agencies["agencies"] = 0

    conn = database_connection("transit")
    existing = conn.execute("SELECT COALESCE(MAX(DISTINCT(agency_id)), 0) FROM agencies;").fetchone()[0]

    transit = create_gtfs_project.new_gtfs_builder(
        agency="Agency_1",
        day="2016-04-13",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"

    transit2 = create_gtfs_project.new_gtfs_builder(
        agency="Agency_2",
        day="2016-07-19",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    transit.save_to_disk()
    transit2.save_to_disk()

    assert conn.execute("SELECT MAX(DISTINCT(agency_id)) FROM agencies;").fetchone()[0] == existing + 2

    transit3 = create_gtfs_project.new_gtfs_builder(
        agency="Agency_3",
        day="2016-07-19",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    transit3.save_to_disk()
    assert conn.execute("SELECT MAX(DISTINCT(agency_id)) FROM agencies;").fetchone()[0] == existing + 3


def test___create_transit_database(create_gtfs_project):
    assert isfile(join(create_gtfs_project.project_base_path, "public_transport.sqlite")) is True
