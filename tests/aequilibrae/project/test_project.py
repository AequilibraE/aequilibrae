import os
from tempfile import gettempdir
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
import pytest

from aequilibrae.utils.create_example import create_example


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
        database_connection(table_type="network")

        project.close()
        with pytest.raises(FileNotFoundError):
            database_connection(table_type="network")

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("routes", "route"),
            ("trips", "pattern_id"),
        ],
    )
    def test_create_empty_transit(self, table: str, exp_column: str):
        fldr = os.path.join(gettempdir(), uuid4().hex)
        proj = create_example(fldr, "nauru")

        proj.create_empty_transit()

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = {x[1] for x in fields}

        assert exp_column in fields, f"Table {table.upper()} was not created correctly"
