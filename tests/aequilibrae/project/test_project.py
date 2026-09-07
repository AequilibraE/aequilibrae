import pytest

from aequilibrae.project.database_connection import database_connection
from aequilibrae.project.project import Project
from aequilibrae.utils.db_utils import read_and_close


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


def test_project_wide_transaction(coquimbo_example):
    assert not (
        coquimbo_example.db_connection.in_transaction
        or coquimbo_example.transit_connection.in_transaction
        or coquimbo_example.results_connection.in_transaction  # Creates results DB
    )

    with coquimbo_example.transaction():
        assert (
            coquimbo_example.db_connection.in_transaction
            and coquimbo_example.transit_connection.in_transaction
            and coquimbo_example.results_connection.in_transaction
        )

    assert not (
        coquimbo_example.db_connection.in_transaction
        or coquimbo_example.transit_connection.in_transaction
        or coquimbo_example.results_connection.in_transaction
    )


def test_cannot_create_db_during_project_wide_transaction(empty_project):
    with empty_project.transaction():
        with pytest.raises(RuntimeError, match="cannot create connection while in collective transaction"):
            # Property access triggers creation
            _ = empty_project.results_connection

        with pytest.raises(RuntimeError, match="cannot create connection while in collective transaction"):
            empty_project.scenario.create_transit_database()

    _ = empty_project.results_connection
    empty_project.scenario.create_transit_database()
