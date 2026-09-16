from os.path import join

import pytest

from aequilibrae.context import activate_project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import read_and_close


def test_cannot_connect_when_no_active_project():
    activate_project(None)
    with pytest.raises(FileNotFoundError):
        database_connection("network")


def test_connection_with_new_project(empty_project):
    with read_and_close(empty_project.path_to_file) as conn:
        links = conn.execute("select count(*) from links").fetchone()[0]
    assert links == 0, "Returned more links thant it should have"


def test_connection_with_transit(empty_project):
    empty_project.scenario.create_transit_database()
    Transit(empty_project)
    with read_and_close(join(empty_project.project_base_path, "public_transport.sqlite")) as conn:
        routes = conn.execute("select count(*) from routes").fetchone()[0]
    assert routes == 0, "Returned more routes thant it should have"


def test_db_connection_is_spatial(empty_project):
    with empty_project.db_connection as conn:
        assert conn.execute("select spatialite_version()").fetchone()[0]


def test_db_connection_spatial_is_deprecated(empty_project):
    with pytest.warns(DeprecationWarning, match="removed in version 2.1"):
        with empty_project.db_connection_spatial as conn:
            assert conn.execute("select spatialite_version()").fetchone()[0]
