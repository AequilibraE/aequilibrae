import sqlite3

import pytest

from aequilibrae.utils.sqlite_utils import without_triggers


@pytest.mark.parametrize("fail", [False, True])
def test_without_triggers_restores_exact_definition(fail):
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            """
            CREATE TABLE source (value INTEGER);
            CREATE TABLE audit (value INTEGER);
            CREATE TRIGGER "aequilibrae_audit ""quoted"" name" AFTER INSERT ON source
            BEGIN
                INSERT INTO audit VALUES (NEW.value);
            END;
            """
        )
        definition = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'").fetchone()
        conn.execute("BEGIN")

        def insert_without_trigger():
            with without_triggers(conn):
                conn.execute("INSERT INTO source VALUES (1)")
                assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 0
                if fail:
                    raise ValueError("operation failed")

        if fail:
            with pytest.raises(ValueError, match="operation failed"):
                insert_without_trigger()
        else:
            insert_without_trigger()

        assert conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'").fetchone() == definition
        conn.execute("INSERT INTO source VALUES (2)")
        assert conn.execute("SELECT value FROM audit").fetchall() == [(2,)]
