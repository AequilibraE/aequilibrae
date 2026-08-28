import json

import pandas as pd
import pytest

from aequilibrae.project.data.results import Results, format_dataframe
from aequilibrae.utils.db_utils import ConnectionClosure


@pytest.fixture
def results():
    closure = ConnectionClosure(":memory:", ":memory:")
    closure.db_connection._connection.execute(
        """CREATE TABLE results (
            scenario TEXT,
            year TEXT,
            table_name TEXT NOT NULL PRIMARY KEY,
            reference_table TEXT,
            procedure TEXT NOT NULL,
            procedure_id TEXT NOT NULL,
            procedure_report TEXT NOT NULL,
            timestamp DATETIME DEFAULT current_timestamp,
            description TEXT
        )"""
    )
    table = Results(closure.db_connection, closure.results_connection)
    yield table, closure.results_connection._connection
    closure.close()


def assignment_frame():
    return pd.DataFrame(
        {"volume_ab": [1250.0, 830.0], "volume_ba": [1100.0, 790.0]},
        index=pd.Index([42, 43], name="link_id"),
    )


def test_create_get_update_and_container_interfaces(results):
    table, _ = results
    data = assignment_frame()

    record = table.create(
        "assignment_2030",
        data,
        procedure="traffic assignment",
        procedure_id="2030-base",
        procedure_report={"converged": True},
        year="2030",
        description="Base assignment",
    )

    assert record == table.get("assignment_2030")
    assert "assignment_2030" in table
    assert len(table) == 1
    assert [item.table_name for item in table] == ["assignment_2030"]
    assert json.loads(record.procedure_report) == {"converged": True}
    assert record.timestamp is not None

    stored = table.get_results(record.table_name)
    pd.testing.assert_frame_equal(stored, data.reset_index())

    table.update(record.table_name, description="Updated")
    assert record.description == "Base assignment"
    assert table.get(record.table_name).description == "Updated"
    assert table.list().loc[0, "table_name"] == record.table_name


def test_create_validates_input_and_does_not_replace_resources(results):
    table, results_connection = results
    table.create("existing", assignment_frame())

    with pytest.raises(ValueError, match="already exists"):
        table.create("existing", assignment_frame())
    with pytest.raises(TypeError, match="pandas DataFrame"):
        table.create("not_a_frame", {"value": [1]})

    colliding_index = pd.DataFrame({"link_id": [1]}, index=pd.Index([1], name="link_id"))
    with pytest.raises(ValueError, match="collides"):
        table.create("collision", colliding_index)

    assert results_connection.execute("SELECT count(*) FROM existing").fetchone()[0] == 2
    assert "collision" not in table


def test_delete_and_delete_result_have_distinct_resource_semantics(results):
    table, results_connection = results
    table.create("metadata_only", assignment_frame())
    table.delete("metadata_only")

    assert "metadata_only" not in table
    assert results_connection.execute("SELECT 1 FROM metadata_only LIMIT 1").fetchone() is not None

    table.update_database()
    assert "metadata_only" in table
    table.delete_result("metadata_only")
    assert "metadata_only" not in table
    assert (
        results_connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata_only'").fetchone()
        is None
    )


def test_clear_update_and_sync_reconcile_metadata_and_data(results):
    table, results_connection = results
    table.insert(table_name="result_with_missing_data")
    results_connection.execute("CREATE TABLE orphan (link_id INTEGER, value REAL)")

    table.sync()

    assert "result_with_missing_data" not in table
    assert "orphan" in table
    assert table.get("orphan").procedure == ""


def test_unnamed_and_multi_indexes_are_saved_as_columns(results):
    table, _ = results
    unnamed = pd.DataFrame({"value": [3, 4]})
    table.create("unnamed", unnamed)
    assert list(table.get_results("unnamed").columns) == ["index_level_0", "value"]

    index = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["link_id", "class"])
    multi = pd.DataFrame({"flow": [1.5, 2.5]}, index=index)
    table.create("multi", multi)
    assert list(table.get_results("multi").columns) == ["link_id", "class", "flow"]


def test_format_dataframe_rejects_ambiguous_columns():
    duplicate_columns = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ValueError, match="unique"):
        format_dataframe(duplicate_columns)

    non_string_columns = pd.DataFrame([[1]], columns=[1])
    with pytest.raises(ValueError, match="strings"):
        format_dataframe(non_string_columns)
