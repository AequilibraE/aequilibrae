from aequilibrae.transit import Transit
from aequilibrae.context import activate_project
from aequilibrae.project.database_connection import database_connection
import pytest


class TestDatabaseConnection:
    def test_cannot_connect_when_no_active_project(self):
        activate_project(None)
        with pytest.raises(FileNotFoundError):
            database_connection("network")

    def test_connection_with_new_project(self, project):
        conn = database_connection(db_type="network", project_path=project.project_base_path)
        cursor = conn.cursor()
        cursor.execute("select count(*) from links")
        assert cursor.fetchone()[0] == 0, "Returned more links thant it should have"

    def test_connection_with_transit(self, project):
        Transit(project)
        conn = database_connection(db_type="transit", project_path=project.project_base_path)
        cursor = conn.cursor()
        cursor.execute("select count(*) from routes")
        assert cursor.fetchone()[0] == 0, "Returned more routes thant it should have"
