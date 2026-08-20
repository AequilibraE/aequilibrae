import sqlite3

import pandas as pd
import pytest

from aequilibrae.project.data.results import Results
from aequilibrae.utils.db_utils import ConnectionClosure


@pytest.fixture
def results():
    closure = ConnectionClosure(sqlite3.connect(":memory:"), sqlite3.connect(":memory:"))
    project = closure.db_connection
    project.connection.execute(
        """CREATE TABLE results (
        table_name TEXT PRIMARY KEY, procedure TEXT, procedure_id TEXT,
        procedure_report TEXT, timestamp TEXT, description TEXT, scenario TEXT,
        year TEXT, reference_table TEXT)"""
    )
    yield Results(project, closure.results_connection), closure
    closure.close()


def test_create_persists_named_index_and_metadata(results):
    table, closure = results
    frame = pd.DataFrame({"flow": [1.5, None]}, index=pd.Index([10, 20], name="link_id"))

    table.create("assignment", frame, procedure="traffic assignment", chunksize=1)

    assert table.get("assignment").procedure == "traffic assignment"
    columns = closure.results_connection.connection.execute("PRAGMA table_info('assignment')").fetchall()
    assert [column[1] for column in columns] == ["link_id", "flow"]
    assert closure.results_connection.connection.execute("SELECT * FROM assignment ORDER BY link_id").fetchall() == [
        (10, 1.5),
        (20, None),
    ]


def test_generic_delete_leaves_data_and_resource_delete_removes_it(results):
    table, closure = results
    frame = pd.DataFrame({"value": [1]}, index=pd.Index([1], name="id"))
    table.create("kept", frame)
    table.delete("kept")
    assert closure.results_connection.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kept'"
    ).fetchone()

    table.insert(table_name="kept")
    table.delete_result("kept")
    assert not table._data_exists("kept")


def test_create_rejects_existing_data_without_replacing_it(results):
    table, closure = results
    closure.results_connection.connection.execute("CREATE TABLE existing (value INTEGER)")
    closure.results_connection.connection.execute("INSERT INTO existing VALUES (7)")

    with pytest.raises(ValueError, match="already exists"):
        table.create("existing", pd.DataFrame({"value": [8]}))

    assert closure.results_connection.connection.execute("SELECT * FROM existing").fetchall() == [(7,)]
    assert "existing" not in table


def test_row_conversion_failure_rolls_back_data_table(results):
    table, closure = results
    frame = pd.DataFrame({"value": [object()]}, index=pd.Index([1], name="id"))

    with pytest.raises(TypeError, match="cannot be stored"):
        table.create("broken", frame)

    assert closure.results_connection.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='broken'"
    ).fetchone() is None
    assert "broken" not in table


def test_resource_helpers_reject_enclosing_transactions(results):
    table, closure = results
    frame = pd.DataFrame({"value": [1]}, index=pd.Index([1], name="id"))
    with closure.transaction():
        with pytest.raises(RuntimeError, match="cannot run"):
            table.create("nested", frame)
