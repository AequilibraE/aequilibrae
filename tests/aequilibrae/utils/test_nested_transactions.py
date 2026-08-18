import sqlite3

import pytest

from aequilibrae.utils.db_utils import ConnectionClosure


def test_nested_failure_rolls_back_only_savepoint():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    manager = closure["project"]
    manager.execute("CREATE TABLE events (name TEXT)")
    try:
        with manager.transaction() as value:
            assert value is None
            manager.execute("INSERT INTO events VALUES ('outer')")
            with pytest.raises(ValueError):
                with manager.transaction():
                    manager.execute("INSERT INTO events VALUES ('inner')")
                    raise ValueError
        assert manager.execute("SELECT name FROM events").fetchall() == [("outer",)]
        assert manager.depth == 0
        assert not manager.in_transaction
    finally:
        closure.close()


def test_closure_rolls_back_every_connection_on_body_error():
    closure = ConnectionClosure({"a": sqlite3.connect(":memory:"), "b": sqlite3.connect(":memory:")})
    try:
        for name in closure:
            closure[name].execute("CREATE TABLE events (name TEXT)")
        with pytest.raises(RuntimeError):
            with closure.transaction() as value:
                assert value is None
                for name in closure:
                    closure[name].execute("INSERT INTO events VALUES ('discarded')")
                raise RuntimeError
        assert all(closure[name].execute("SELECT * FROM events").fetchall() == [] for name in closure)
    finally:
        closure.close()


def test_deferred_constraint_commit_failure_leaves_manager_usable():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    manager = closure["project"]
    manager.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
    manager.execute(
        "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED)"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with manager.transaction():
                manager.execute("INSERT INTO child VALUES (1)")
        assert not manager.in_transaction
        with manager.transaction():
            manager.execute("INSERT INTO parent VALUES (1)")
    finally:
        closure.close()


def test_closure_validation_is_all_or_nothing():
    first = sqlite3.connect(":memory:")
    second = sqlite3.connect(":memory:")
    first.isolation_level = "IMMEDIATE"
    second.execute("BEGIN")
    try:
        with pytest.raises(ValueError):
            ConnectionClosure({"first": first, "second": second})
        assert first.isolation_level == "IMMEDIATE"
    finally:
        second.rollback()
        first.close()
        second.close()


def test_manager_does_not_expose_finalization_or_raw_connection():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    try:
        manager = closure["project"]
        for attribute in ("commit", "rollback", "close", "connection"):
            assert not hasattr(manager, attribute)
        assert manager.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        closure.close()


def test_teardown_rejects_active_transaction():
    closure = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    transaction = closure["project"].transaction()
    transaction.__enter__()
    try:
        with pytest.raises(RuntimeError):
            closure.close()
    finally:
        transaction.__exit__(RuntimeError, RuntimeError(), None)
        closure.close()
