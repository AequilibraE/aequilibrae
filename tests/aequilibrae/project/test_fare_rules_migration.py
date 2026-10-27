import importlib.util
import sqlite3
from pathlib import Path

import aequilibrae

MIGRATION = (
    Path(aequilibrae.__file__).parent
    / "project"
    / "database_specification"
    / "transit"
    / "migrations"
    / "008_fare_rule_zones_as_text.py"
)

OLD_FARE_RULES = """create TABLE fare_rules (fare_id INTEGER NOT NULL, route_id INTEGER, origin TEXT, destination TEXT,
                                         contains INTEGER)"""


def migrate(conn: sqlite3.Connection) -> None:
    spec = importlib.util.spec_from_file_location("fare_rule_zones_as_text", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.migrate(project_conn=None, transit_conn=conn, results_conn=None)


def test_migration_stores_fare_rule_zones_as_text():
    """Test that migration 008 declares fare rule zones as text and keeps existing rules."""
    conn = sqlite3.connect(":memory:")
    conn.execute(OLD_FARE_RULES)
    rows = [(1, 10, "A", "B", "Z1"), (2, None, "07", None, "07"), (3, 11, None, None, None)]
    conn.executemany("INSERT INTO fare_rules VALUES (?, ?, ?, ?, ?)", rows)

    migrate(conn)

    declared = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(fare_rules)")}
    assert declared["contains"] == "TEXT"

    stored = conn.execute("SELECT fare_id, route_id, origin, destination, contains FROM fare_rules ORDER BY fare_id")
    assert stored.fetchall() == [(1, 10, "A", "B", "Z1"), (2, None, "07", None, "7"), (3, 11, None, None, None)]

    conn.execute("INSERT INTO fare_rules (fare_id, contains) VALUES (4, '07')")
    assert conn.execute("SELECT contains FROM fare_rules WHERE fare_id = 4").fetchone() == ("07",)


def test_migration_skips_databases_without_fare_rules():
    """Test that migration 008 leaves a transit database without fare rules unchanged."""
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_new_transit_databases_store_fare_rule_zones_as_text(empty_project):
    """Test that new transit databases declare fare rule zones as text."""
    empty_project.scenario.create_transit_database()
    with empty_project.transit_connection as conn:
        declared = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(fare_rules)")}
    assert declared["contains"] == "TEXT"
