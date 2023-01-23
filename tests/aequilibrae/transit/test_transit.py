from os.path import isfile, join
from aequilibrae.transit import Transit


def test_new_gtfs_builder(create_gtfs_project, create_path):
    transit = create_gtfs_project.new_gtfs_builder(
        agency="LISERCO, LISANCO, LINCOSUR",
        file_path=join(create_path, "gtfs_coquimbo.zip"),
    )

    assert str(type(transit)) == "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>"


def test___create_transit_database(create_gtfs_project):

    assert isfile(join(create_gtfs_project.project_base_path, "public_transport.sqlite")) is True
