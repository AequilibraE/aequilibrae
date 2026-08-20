import sqlite3
from dataclasses import fields

import pandas as pd
import pytest

from aequilibrae.project.project_table import NonSpatialProjectTable
from aequilibrae.utils.db_utils import ConnectionClosure


class Things(NonSpatialProjectTable):
    name = "things"
    key = "thing_id"
    record_name = "ThingRecord"


@pytest.fixture
def things():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    transactions = closure["project"]
    transactions.execute("CREATE TABLE things (thing_id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
    yield Things(transactions), transactions
    closure.close()


def test_standalone_writes_commit_and_data_includes_key_column(things):
    table, transactions = things
    table.insert(thing_id=1, value=10)

    assert not transactions.in_transaction
    assert table.get(1).value == 10
    assert "thing_id" in table.data.columns
    assert table.data.loc[0, "thing_id"] == 1


def test_guessed_records_refresh_after_user_fields_are_added(things):
    table, transactions = things
    transactions.execute("ALTER TABLE things ADD COLUMN user_note TEXT")
    table.insert(thing_id=1, value=10, user_note="custom")

    record = table.get(1)
    assert record.thing_id == 1
    assert record.value == 10
    assert type(record).__name__ == "ThingRecord"
    assert [field.name for field in fields(record)] == ["thing_id", "value", "user_note"]
    assert type(record).__annotations__ == {"thing_id": int, "value": int, "user_note": str | None}
    assert record.user_note == "custom"
    assert table.data.loc[0, "user_note"] == "custom"


def test_gateway_mutation_is_a_savepoint(things):
    table, transactions = things
    table.insert(thing_id=1, value=10)

    with transactions.transaction():
        table.update(1, value=20)
        try:
            with transactions.transaction():
                table.update(1, value=30)
                raise ValueError("discard nested scope")
        except ValueError:
            pass
        assert table.get(1).value == 20

    assert table.get(1).value == 20


def test_update_from_uses_the_key_column(things):
    table, _ = things
    table.insert_from(pd.DataFrame({"thing_id": [1, 2], "value": [10, 20]}))
    updates = pd.DataFrame({"thing_id": [1, 2], "value": [11, 21]})
    original = updates.copy()

    table.update_from(updates)

    pd.testing.assert_frame_equal(updates, original)
    assert [table.get(key).value for key in updates.thing_id] == [11, 21]


def test_failing_bulk_update_is_atomic(things):
    table, _ = things
    table.insert_from(pd.DataFrame({"thing_id": [1, 2], "value": [10, 20]}))
    updates = pd.DataFrame({"thing_id": [1, 999], "value": [11, 21]})

    with pytest.raises(ValueError, match="999"):
        table.update_from(updates)

    assert table.get(1).value == 10
