import sqlite3
from pathlib import Path

import pytest

from aequilibrae.utils.model_run_utils import import_file_as_module


MIGRATIONS = Path(__file__).parents[3] / "src" / "aequilibrae" / "project" / "database_specification"
NETWORK_MIGRATIONS = sorted((MIGRATIONS / "network" / "migrations").glob("[0-9][0-9][0-9]_*.py"))
TRANSIT_MIGRATIONS = sorted((MIGRATIONS / "transit" / "migrations").glob("[0-9][0-9][0-9]_*.py"))


def _migrate_function(path: Path):
    return import_file_as_module(path, f"test_migration_{path.parent.parent.name}_{path.stem}", force=True).migrate


@pytest.mark.parametrize("migration_file", NETWORK_MIGRATIONS)
def test_network_migrations_require_project_connection(migration_file):
    migrate = _migrate_function(migration_file)

    with pytest.raises(RuntimeError, match="requires a project_conn connection"):
        migrate(project_conn=None, transit_conn=None, results_conn=None)


@pytest.mark.parametrize("migration_file", TRANSIT_MIGRATIONS)
def test_transit_migrations_require_transit_connection(migration_file):
    migrate = _migrate_function(migration_file)
    project_conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="requires a transit_conn connection"):
            migrate(project_conn=project_conn, transit_conn=None, results_conn=None)
    finally:
        project_conn.close()


def test_transit_results_migration_requires_results_connection_when_needed():
    migrate = _migrate_function(MIGRATIONS / "transit" / "migrations" / "004_move_results_to_project.py")
    project_conn = sqlite3.connect(":memory:")
    transit_conn = sqlite3.connect(":memory:")
    try:
        transit_conn.execute("CREATE TABLE results (table_name TEXT)")
        with pytest.raises(RuntimeError, match="requires a results_conn connection"):
            migrate(project_conn=project_conn, transit_conn=transit_conn, results_conn=None)
    finally:
        transit_conn.close()
        project_conn.close()
