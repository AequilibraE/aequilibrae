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
    gateway, closure = results
    frame = pd.DataFrame({"flow": [1.5, None]}, index=pd.Index([10, 20], name="link_id"))

    gateway.create("assignment", frame, procedure="traffic assignment", chunksize=1)

    assert gateway.get("assignment").procedure == "traffic assignment"
    columns = closure.results_connection.connection.execute("PRAGMA table_info('assignment')").fetchall()
    assert [column[1] for column in columns] == ["link_id", "flow"]
    assert closure.results_connection.connection.execute("SELECT * FROM assignment ORDER BY link_id").fetchall() == [
        (10, 1.5),
        (20, None),
    ]


def test_generic_delete_leaves_payload_and_resource_delete_removes_it(results):
    gateway, closure = results
    frame = pd.DataFrame({"value": [1]}, index=pd.Index([1], name="id"))
    gateway.create("kept", frame)
    gateway.delete("kept")
    assert closure.results_connection.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kept'"
    ).fetchone()

    gateway.insert(table_name="kept")
    gateway.delete_result("kept")
    assert not gateway._payload_exists("kept")


def test_create_rejects_existing_payload_without_replacing_it(results):
    gateway, closure = results
    closure.results_connection.connection.execute("CREATE TABLE existing (value INTEGER)")
    closure.results_connection.connection.execute("INSERT INTO existing VALUES (7)")

    with pytest.raises(ValueError, match="already exists"):
        gateway.create("existing", pd.DataFrame({"value": [8]}))

    assert closure.results_connection.connection.execute("SELECT * FROM existing").fetchall() == [(7,)]
    assert "existing" not in gateway


def test_row_conversion_failure_rolls_back_payload_table(results):
    gateway, closure = results
    frame = pd.DataFrame({"value": [object()]}, index=pd.Index([1], name="id"))

    with pytest.raises(TypeError, match="cannot be stored"):
        gateway.create("broken", frame)

    assert closure.results_connection.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='broken'"
    ).fetchone() is None
    assert "broken" not in gateway


def test_resource_helpers_reject_enclosing_transactions(results):
    gateway, closure = results
    frame = pd.DataFrame({"value": [1]}, index=pd.Index([1], name="id"))
    with closure.transaction():
        with pytest.raises(RuntimeError, match="cannot run"):
            gateway.create("nested", frame)
