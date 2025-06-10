from os.path import join

from aequilibrae.transit import Transit
from aequilibrae.context import activate_project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.utils.db_utils import read_and_close
import pytest


class TestDatabaseConnection:
    def test_cannot_connect_when_no_active_project(self):
        activate_project(None)
        with pytest.raises(FileNotFoundError):
            database_connection("network")

    def test_connection_with_new_project(self, project):
        with read_and_close(project.path_to_file) as conn:
            links = conn.execute("select count(*) from links").fetchone()[0]
        assert links == 0, "Returned more links thant it should have"

    def test_connection_with_transit(self, project):
        Transit(project)

        with read_and_close(join(project.project_base_path, "public_transport.sqlite")) as conn:
            routes = conn.execute("select count(*) from routes").fetchone()[0]
        assert routes == 0, "Returned more routes thant it should have"
