import sqlite3

import pytest

from aequilibrae.project.tools import MigrationManager, MigrationStatus
from aequilibrae.utils.db_utils import AequilibraEConnection


@pytest.fixture
def connections():
    conn_dict = {"conn": sqlite3.connect(":memory:", factory=AequilibraEConnection)}
    yield conn_dict
    conn_dict["conn"].close()


@pytest.fixture
def main_connection():
    return "conn"


@pytest.fixture
def migrations_file(test_data_path):
    return test_data_path / "mock_migrations" / "init.py"


@pytest.fixture
def migrations_duplicate(test_data_path):
    return test_data_path / "mock_migrations" / "duplicate_init.py"


@pytest.fixture
def migrations_negative(test_data_path):
    return test_data_path / "mock_migrations" / "negative_init.py"


def test_migration_manager_init(migrations_file):
    manager = MigrationManager(migrations_file)

    # Check migrations were loaded correctly
    assert len(manager.migrations) == 6
    assert manager.migrations[0].name == "initial_migration"
    assert manager.migrations[1].name == "add_users"
    assert manager.migrations[2].name == "add_posts"
    assert manager.migrations[3].name == "add_comments"
    assert manager.migrations[4].name == "invalid_migration"
    assert manager.migrations[5].name == "non_callable_migrate"


def test_migration_manager_duplicate_ids(migrations_duplicate):
    with pytest.raises(ValueError):
        MigrationManager(migrations_duplicate)


def test_migration_manager_invalid_id(migrations_negative):
    with pytest.raises(ValueError):
        MigrationManager(migrations_negative)


def test_status(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    conn = connections[main_connection]

    # Initially all should be missing except initial which gets auto-applied
    status = manager.status(conn)
    assert status[0] == MigrationStatus.APPLIED
    assert status[1] == MigrationStatus.MISSING
    assert status[2] == MigrationStatus.MISSING
    assert status[3] == MigrationStatus.MISSING

    # Check migrations table was created
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
    assert conn.execute(sql).fetchone() is not None


def test_mark_all_as_seen(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    conn = connections[main_connection]
    manager.mark_all_as_seen(conn)

    # All should be marked as MISSING
    status = manager.status(conn)
    for id, stat in status.items():
        if id == 0:
            assert stat == MigrationStatus.APPLIED
        else:
            assert stat == MigrationStatus.MISSING

    # Check entries exist in migrations table
    rows = conn.execute("SELECT id, status FROM migrations ORDER BY id").fetchall()
    assert len(rows) == 6
    assert rows[0][1] == "APPLIED"
    assert rows[1][1] == "MISSING"
    assert rows[2][1] == "MISSING"
    assert rows[3][1] == "MISSING"
    assert rows[4][1] == "MISSING"
    assert rows[5][1] == "MISSING"


def test_find_applicable(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    conn = connections[main_connection]

    # Should find all non-initial migrations
    applicable = manager.find_applicable(conn)
    assert len(applicable) == 5
    assert applicable[0].id == 1
    assert applicable[1].id == 2
    assert applicable[2].id == 3
    assert applicable[3].id == 4
    assert applicable[4].id == 5

    # Apply first and second migration
    applicable[0].apply(conn, connections)
    applicable[0].mark_as(conn, MigrationStatus.APPLIED)

    applicable[1].apply(conn, connections)
    applicable[1].mark_as(conn, MigrationStatus.APPLIED)

    # Should now find only remaining migrations
    applicable = manager.find_applicable(conn)
    assert len(applicable) == 3
    assert applicable[0].id == 3
    assert applicable[1].id == 4
    assert applicable[2].id == 5


def test_out_of_order_migrations(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    conn = connections[main_connection]

    # Apply migrations 0, 1, and 3 but not 2
    manager.migrations[0].apply(conn, connections)
    manager.migrations[0].mark_as(conn, MigrationStatus.APPLIED)

    manager.migrations[1].apply(conn, connections)
    manager.migrations[1].mark_as(conn, MigrationStatus.APPLIED)

    manager.migrations[3].apply(conn, connections)
    manager.migrations[3].mark_as(conn, MigrationStatus.APPLIED)

    # Should raise error because migration 2 was skipped
    with pytest.raises(RuntimeError):
        manager.find_applicable(conn)


def test_upgrade(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    del manager.migrations[4]  # drop the duds
    del manager.migrations[5]

    # Upgrade should apply all migrations
    manager.upgrade(main_connection, connections)

    # Check all migrations were applied
    status = manager.status(connections[main_connection])
    for _id, stat in status.items():
        assert stat == MigrationStatus.APPLIED

    # Check tables were created
    tables = connections[main_connection].execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "migrations" in table_names
    assert "users" in table_names
    assert "posts" in table_names
    assert "comments" in table_names


def test_upgrade_with_skip(migrations_file, connections, main_connection):
    manager = MigrationManager(migrations_file)
    conn = connections[main_connection]
    del manager.migrations[4]
    del manager.migrations[5]

    manager.mark_all_as_seen(conn)

    # Skip migration 2
    manager.upgrade(main_connection, connections, skip={2})

    # Check migrations 1 and 3 were applied, 2 was skipped
    status = manager.status(conn)
    assert status[0] == MigrationStatus.APPLIED
    assert status[1] == MigrationStatus.APPLIED
    assert status[2] == MigrationStatus.SKIPPED
    assert status[3] == MigrationStatus.APPLIED

    # There are no applicable upgrades now
    applicable = manager.find_applicable(conn)
    assert applicable == []

    # Check tables were created (should have users and comments but not posts)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "migrations" in table_names
    assert "users" in table_names
    assert "posts" not in table_names  # Was skipped
    assert "comments" in table_names

    manager.migrations[2].apply(conn, connections)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert "posts" in [t[0] for t in tables]  # Was just applied

    status = manager.status(conn)
    assert status[2] == MigrationStatus.APPLIED
