import sqlite3
from pathlib import Path

import pytest

from aequilibrae.project.tools import MigrationManager, MigrationStatus
from aequilibrae.project.tools.migration_manager import iter_sql_statements
from aequilibrae.utils.db_utils import ConnectionClosure


@pytest.fixture
def closure():
    owner = ConnectionClosure({"project": sqlite3.connect(":memory:")})
    yield owner
    owner.close()


@pytest.fixture
def migrations_file(test_data_path):
    return test_data_path / "mock_migrations" / "init.py"


def test_migration_manager_init(migrations_file):
    manager = MigrationManager(migrations_file)
    assert list(manager.migrations) == [0, 1, 2, 3, 4, 5]
    assert manager.migrations[0].type == "sql"
    assert manager.migrations[3].type == "py"


def test_migration_manager_rejects_duplicate_and_negative_ids(test_data_path):
    with pytest.raises(ValueError):
        MigrationManager(test_data_path / "mock_migrations" / "duplicate_init.py")
    with pytest.raises(ValueError):
        MigrationManager(test_data_path / "mock_migrations" / "negative_init.py")


def test_status_initializes_migration_table(migrations_file, closure):
    manager = MigrationManager(migrations_file)
    status = manager.status(closure)
    assert status[0] == MigrationStatus.APPLIED
    assert all(value == MigrationStatus.MISSING for key, value in status.items() if key)


def test_upgrade_applies_schema_and_status_together(migrations_file, closure):
    manager = MigrationManager(migrations_file)
    del manager.migrations[4]
    del manager.migrations[5]

    manager.upgrade(closure)

    assert set(manager.status(closure).values()) == {MigrationStatus.APPLIED}
    tables = {
        row[0]
        for row in closure["project"].execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"migrations", "users", "posts", "comments"} <= tables


def test_failing_sql_rolls_back_schema_prefix_and_status(tmp_path, closure):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "000_initial.sql").write_text(
        "CREATE TABLE migrations (id INTEGER PRIMARY KEY, name TEXT, status TEXT, date TEXT);"
    )
    (migration_dir / "001_failure.sql").write_text(
        "CREATE TABLE prefix (id INTEGER); INSERT INTO table_that_does_not_exist VALUES (1);"
    )
    (migration_dir / "migrations.py").write_text(
        "from pathlib import Path\n"
        "path = Path(__file__).parent\n"
        "migrations = [path / '000_initial.sql', path / '001_failure.sql']\n"
    )
    manager = MigrationManager(migration_dir / "migrations.py")

    with pytest.raises(sqlite3.OperationalError):
        manager.upgrade(closure)

    assert closure["project"].execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prefix'"
    ).fetchone() is None
    assert closure["project"].execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations'"
    ).fetchone() is None


def test_sql_statement_iterator_preserves_trigger_bodies():
    sql = """
    CREATE TABLE events (id INTEGER);
    CREATE TABLE audit (id INTEGER);
    CREATE TRIGGER log_event AFTER INSERT ON events BEGIN
      INSERT INTO audit VALUES (NEW.id);
      INSERT INTO audit VALUES (NEW.id + 1);
    END;
    """
    statements = list(iter_sql_statements(sql))
    assert len(statements) == 3
    assert "NEW.id + 1" in statements[-1]


def test_sql_statement_iterator_rejects_incomplete_tail():
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_sql_statements("CREATE TABLE incomplete (id INTEGER)"))


def test_all_historical_sql_migrations_are_complete():
    root = Path(__file__).parents[3] / "src" / "aequilibrae" / "project" / "database_specification"
    for path in root.glob("*/migrations/*.sql"):
        assert list(iter_sql_statements(path.read_text())), path
