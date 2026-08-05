import pytest

from aequilibrae.project.tools import MigrationManager, MigrationStatus
from aequilibrae.utils.db_utils import ConnectionClosure


@pytest.fixture
def closure():
    owner = ConnectionClosure(":memory:")
    yield owner
    owner.close()


@pytest.fixture
def migrations_file(test_data_path):
    return test_data_path / "mock_migrations" / "init.py"


@pytest.fixture
def migrations_duplicate(test_data_path):
    return test_data_path / "mock_migrations" / "duplicate_init.py"


@pytest.fixture
def migrations_negative(test_data_path):
    return test_data_path / "mock_migrations" / "negative_init.py"


def _apply(manager, migration_id, closure):
    with closure.db_connection.transaction() as conn:
        manager.migrations[migration_id].apply(conn, manager._connections(closure))


def test_migration_manager_init(migrations_file):
    manager = MigrationManager(migrations_file)

    # Check migrations were loaded correctly
    assert list(manager.migrations) == [0, 1, 2, 3, 4, 5]
    assert manager.migrations[0].name == "initial_migration"
    assert manager.migrations[1].name == "add_users"
    assert manager.migrations[2].name == "add_posts"
    assert manager.migrations[3].name == "add_comments"
    assert manager.migrations[4].name == "invalid_migration"
    assert manager.migrations[5].name == "non_callable_migrate"
    assert manager.database == "project"


def test_migration_manager_duplicate_ids(migrations_duplicate):
    with pytest.raises(ValueError):
        MigrationManager(migrations_duplicate)


def test_migration_manager_invalid_id(migrations_negative):
    with pytest.raises(ValueError):
        MigrationManager(migrations_negative)


def test_status(migrations_file, closure):
    manager = MigrationManager(migrations_file)

    # Initially all should be missing except initial which gets auto-applied
    status = manager.status(closure)
    assert status[0] == MigrationStatus.APPLIED
    assert status[1] == MigrationStatus.MISSING
    assert status[2] == MigrationStatus.MISSING
    assert status[3] == MigrationStatus.MISSING

    # Check migrations table was created
    assert closure.db_connection._connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
    ).fetchone() is not None


def test_mark_all_as_seen(migrations_file, closure):
    manager = MigrationManager(migrations_file)
    manager.mark_all_as_seen(closure)

    status = manager.status(closure)
    for id_, stat in status.items():
        if id_ == 0:
            assert stat == MigrationStatus.APPLIED
        else:
            assert stat == MigrationStatus.MISSING

    # Check entries exist in migrations table
    rows = closure.db_connection._connection.execute("SELECT id, status FROM migrations ORDER BY id").fetchall()
    assert len(rows) == 6
    assert rows[0][1] == "APPLIED"
    assert rows[1][1] == "MISSING"
    assert rows[2][1] == "MISSING"
    assert rows[3][1] == "MISSING"
    assert rows[4][1] == "MISSING"
    assert rows[5][1] == "MISSING"


def test_find_applicable(migrations_file, closure):
    manager = MigrationManager(migrations_file)

    # Should find all non-initial migrations
    applicable = manager.find_applicable(closure)
    assert [migration.id for migration in applicable] == [1, 2, 3, 4, 5]

    # Apply the first two migrations
    _apply(manager, 1, closure)
    _apply(manager, 2, closure)

    applicable = manager.find_applicable(closure)
    assert [migration.id for migration in applicable] == [3, 4, 5]


def test_invalid_migration_callable(migrations_file, closure):
    manager = MigrationManager(migrations_file)

    # Migration 4 does not expose a ``migrate`` function
    with pytest.raises(RuntimeError, match="does not expose a global 'migrate' callable"):
        _apply(manager, 4, closure)


def test_non_callable_migrate(migrations_file, closure):
    manager = MigrationManager(migrations_file)

    # Migration 5 exposes a ``migrate`` symbol that is not callable
    with pytest.raises(RuntimeError, match="not callable"):
        _apply(manager, 5, closure)


def test_out_of_order_migrations(migrations_file, closure):
    manager = MigrationManager(migrations_file)

    # Apply migrations 0, 1, and 3 but not 2
    _apply(manager, 0, closure)
    _apply(manager, 1, closure)
    _apply(manager, 3, closure)

    # Should raise error because migration 2 was skipped
    with pytest.raises(RuntimeError):
        manager.find_applicable(closure)


def test_upgrade(migrations_file, closure):
    manager = MigrationManager(migrations_file)
    del manager.migrations[4]  # drop the duds
    del manager.migrations[5]

    # Upgrade should apply all migrations
    manager.upgrade(closure)

    # Check all migrations were applied
    status = manager.status(closure)
    assert all(stat == MigrationStatus.APPLIED for stat in status.values())

    # Check tables were created
    tables = {
        row[0]
        for row in closure.db_connection._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"migrations", "users", "posts", "comments"} <= tables


def test_upgrade_with_skip(migrations_file, closure):
    manager = MigrationManager(migrations_file)
    del manager.migrations[4]
    del manager.migrations[5]

    manager.mark_all_as_seen(closure)

    # Skip migration 2
    manager.upgrade(closure, skip={2})

    # Check migrations 1 and 3 were applied, 2 was skipped
    status = manager.status(closure)
    assert status[0] == MigrationStatus.APPLIED
    assert status[1] == MigrationStatus.APPLIED
    assert status[2] == MigrationStatus.SKIPPED
    assert status[3] == MigrationStatus.APPLIED

    # There are no applicable upgrades now
    assert manager.find_applicable(closure) == []

    # Check tables were created (should have users and comments but not posts)
    tables = {
        row[0]
        for row in closure.db_connection._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "migrations" in tables
    assert "users" in tables
    assert "posts" not in tables  # Was skipped
    assert "comments" in tables

    _apply(manager, 2, closure)

    tables = {
        row[0]
        for row in closure.db_connection._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "posts" in tables  # Was just applied

    assert manager.status(closure)[2] == MigrationStatus.APPLIED
