from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
import pytest


class TestProject:
    def test_opening_wrong_folder(self, tmp_path):
        not_a_project = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            proj = Project()
            proj.open(not_a_project)

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("links", "distance"),
            ("nodes", "is_centroid"),
        ],
    )
    def test_table_creation(self, table: str, exp_column: str, project: Project):
        curr = project.conn.cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = {x[1] for x in fields}

        assert exp_column in fields, f"Table {table.upper()} was not created correctly"

    def test_close(self, project):
        database_connection(db_type="network")

        project.close()
        with pytest.raises(FileNotFoundError):
            database_connection(db_type="network")
