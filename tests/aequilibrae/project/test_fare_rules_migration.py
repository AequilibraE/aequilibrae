import importlib.util
import sqlite3
from pathlib import Path

import aequilibrae

MIGRATION = Path(aequilibrae.__file__).parent / "project/database_specification/transit/migrations"


def test_migration_stores_fare_rule_zones_as_text():
    """Test that migration 008 declares fare rule zones as text and keeps existing rules."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fare_rules (fare_id INTEGER, route_id INTEGER, origin, destination, contains INTEGER)")
    rows = [(1, 10, "A", "B", "Z1"), (2, None, "07", None, 7)]
    conn.executemany("INSERT INTO fare_rules VALUES (?, ?, ?, ?, ?)", rows)

    spec = importlib.util.spec_from_file_location("migration", MIGRATION / "008_fare_rule_zones_as_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.migrate(project_conn=None, transit_conn=conn, results_conn=None)

    conn.execute("INSERT INTO fare_rules (fare_id, contains) VALUES (3, '07')")
    rows = conn.execute("SELECT fare_id, route_id, origin, destination, contains FROM fare_rules ORDER BY fare_id")
    assert rows.fetchall() == [(1, 10, "A", "B", "Z1"), (2, None, "07", None, "7"), (3, None, None, None, "07")]
