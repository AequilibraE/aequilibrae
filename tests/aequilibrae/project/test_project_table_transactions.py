import sqlite3

import pandas as pd
import pytest

from aequilibrae.project.project_table import ProjectTable
from aequilibrae.utils.db_utils import ConnectionClosure


class Things(ProjectTable):
    name = "things"
    key = "thing_id"


@pytest.fixture
def things():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    transactions = closure["project"]
    transactions.execute("CREATE TABLE things (thing_id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
    yield Things(transactions), transactions
    closure.close()


def test_standalone_writes_commit_and_data_uses_key_index(things):
    table, transactions = things
    table.insert(thing_id=1, value=10)

    assert not transactions.in_transaction
    assert table.get(1).value == 10
    assert table.data.index.name == "thing_id"
    assert "thing_id" not in table.data.columns


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


def test_update_from_requires_and_preserves_named_unique_index(things):
    table, _ = things
    table.insert_from(pd.DataFrame({"thing_id": [1, 2], "value": [10, 20]}))
    updates = pd.DataFrame({"value": [11, 21]}, index=pd.Index([1, 2], name="thing_id"))

    table.update_from(updates)

    assert updates.index.equals(pd.Index([1, 2], name="thing_id"))
    assert [table.get(key).value for key in updates.index] == [11, 21]
    with pytest.raises(ValueError, match="must be named"):
        table.update_from(updates.rename_axis(None))
    with pytest.raises(ValueError, match="unique"):
        table.update_from(pd.DataFrame({"value": [1, 2]}, index=pd.Index([1, 1], name="thing_id")))


def test_failing_bulk_update_is_atomic(things):
    table, _ = things
    table.insert_from(pd.DataFrame({"thing_id": [1, 2], "value": [10, 20]}))
    updates = pd.DataFrame({"value": [11, 21]}, index=pd.Index([1, 999], name="thing_id"))

    with pytest.raises(ValueError, match="999"):
        table.update_from(updates)

    assert table.get(1).value == 10
