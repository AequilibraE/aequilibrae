import sqlite3

import pytest

from aequilibrae.utils.db_utils import ConnectionClosure, NestedTransactionManager


def test_nested_failure_rolls_back_only_savepoint():
    closure = ConnectionClosure(":memory:")
    manager = closure.db_connection
    manager._connection.execute("CREATE TABLE events (name TEXT)")
    try:
        with manager.transaction() as connection:
            assert isinstance(connection, sqlite3.Connection)
            connection.execute("INSERT INTO events VALUES ('outer')")
            with pytest.raises(ValueError):
                with manager.transaction() as nested_connection:
                    nested_connection.execute("INSERT INTO events VALUES ('inner')")
                    raise ValueError
        assert manager._connection.execute("SELECT name FROM events").fetchall() == [("outer",)]
        assert manager.depth == 0
        assert not manager.in_transaction
    finally:
        closure.close()


def test_closure_rolls_back_every_connection_on_body_error():
    closure = ConnectionClosure(":memory:", ":memory:", ":memory:")
    managers = (closure.db_connection, closure.results_connection, closure.transit_connection)
    try:
        for manager in managers:
            manager._connection.execute("CREATE TABLE events (name TEXT)")
        with pytest.raises(RuntimeError):
            with closure.transaction():
                for manager in managers:
                    manager._connection.execute("INSERT INTO events VALUES ('discarded')")
                raise RuntimeError
        assert all(manager._connection.execute("SELECT * FROM events").fetchall() == [] for manager in managers)
    finally:
        closure.close()


def test_deferred_constraint_commit_failure_leaves_sqlite_to_recover():
    closure = ConnectionClosure(":memory:")
    manager = closure.db_connection
    connection = manager._connection
    connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
    connection.execute("CREATE TABLE child (parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED)")
    assert not manager.in_transaction
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with manager.transaction() as connection:
                connection.execute("INSERT INTO child VALUES (1)")
        with manager.transaction() as connection:
            connection.execute("INSERT INTO parent VALUES (1)")
    finally:
        closure.close()


def test_closure_rejects_non_path_slots():
    with pytest.raises(TypeError, match="must be paths"):
        ConnectionClosure(object())


def test_optional_connection_properties_are_descriptive():
    closure = ConnectionClosure(":memory:")
    try:
        with pytest.raises(RuntimeError, match="no results database"):
            _ = closure.results_connection
        with pytest.raises(RuntimeError, match="no transit database"):
            _ = closure.transit_connection
    finally:
        closure.close()


def test_context_manager_without_nested_transactions_stays_in_original_transaction():
    closure = ConnectionClosure(":memory:")
    assert closure.db_connection.depth == 0

    with closure.db_connection.transaction() as connection:
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        assert closure.db_connection.depth == 1

    assert closure.db_connection.depth == 0
    try:
        assert not closure.db_connection.in_transaction
        with closure.db_connection.transaction() as connection:
            assert closure.db_connection.in_transaction and closure.db_connection.depth == 1
            transaction_id = closure.db_connection.transaction_id
            connection.execute("INSERT INTO parent VALUES (1)")

            with closure.db_connection as connection:
                assert closure.db_connection.transaction_id == transaction_id and closure.db_connection.depth == 2
                connection.execute("INSERT INTO parent VALUES (2)")

            assert closure.db_connection.transaction_id == transaction_id and closure.db_connection.depth == 1
    finally:
        closure.close()


def test_open_close_manager_only_holds_connection_during_transactions(tmp_path):
    database = tmp_path / "open-close.sqlite"
    database.touch()
    used_connections = []
    manager = NestedTransactionManager(database, open_close=True)
    with pytest.raises(RuntimeError, match="closed outside a transaction"):
        _ = manager._connection

    with manager.transaction() as connection:
        used_connections.append(connection)
        connection.execute("CREATE TABLE events (name TEXT)")
        with manager as nested_connection:
            assert nested_connection is connection
            nested_connection.execute("INSERT INTO events VALUES ('kept')")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        used_connections[-1].execute("SELECT 1")

    with manager as connection:
        used_connections.append(connection)
        assert connection.execute("SELECT name FROM events").fetchall() == [("kept",)]

    with pytest.raises(ValueError, match="discard transaction"):
        with manager as connection:
            used_connections.append(connection)
            connection.execute("INSERT INTO events VALUES ('discarded')")
            raise ValueError("discard transaction")

    with manager as connection:
        used_connections.append(connection)
        assert connection.execute("SELECT name FROM events").fetchall() == [("kept",)]

    assert len({id(connection) for connection in used_connections}) == 4
    manager.close()


def test_spatial_flag_uses_spatialite_connection(tmp_path):
    database = tmp_path / "spatial.sqlite"
    database.touch()
    manager = NestedTransactionManager(database, spatial=True)
    try:
        assert manager._connection.execute("SELECT spatialite_version()").fetchone()[0]
    finally:
        manager.close()


def test_empty_stack_contect_manager_without_transaction_starts_one():
    closure = ConnectionClosure(":memory:")

    try:
        assert not closure.db_connection.in_transaction
        with closure.db_connection as connection:
            assert closure.db_connection.in_transaction
            connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        assert not closure.db_connection.in_transaction
    finally:
        closure.close()
