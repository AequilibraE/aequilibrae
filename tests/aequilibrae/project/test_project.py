from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.utils.db_utils import read_and_close
import pytest


def test_opening_wrong_folder(tmp_path):
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
def test_table_creation(table: str, exp_column: str, empty_project):
    with read_and_close(empty_project.path_to_file) as conn:
        fields = {x[1] for x in conn.execute(f"PRAGMA table_info({table});").fetchall()}

    assert exp_column in fields, f"Table {table.upper()} was not created correctly"


def test_close(empty_no_triggers_project):
    database_connection(db_type="network")

    empty_no_triggers_project.close()
    with pytest.raises(FileNotFoundError):
        database_connection(db_type="network")
