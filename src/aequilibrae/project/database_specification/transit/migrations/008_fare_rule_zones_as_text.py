import sqlite3
from typing import Optional

FARE_RULES_SQL = """CREATE TABLE fare_rules (
	fare_id     INTEGER  NOT NULL,
	route_id    INTEGER,
	origin      TEXT,
	destination TEXT,
	contains    TEXT,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred
);"""


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Declare the zones that fare rules contain as text, as GTFS zone IDs are text"""
    if transit_conn is None:
        raise RuntimeError("Transit migration 008 requires a transit_conn connection")

    if not transit_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fare_rules'").fetchone():
        return

    sql = 'SELECT fare_id, route_id, CAST(origin AS TEXT), CAST(destination AS TEXT), CAST("contains" AS TEXT)'
    rows = transit_conn.execute(f"{sql} FROM fare_rules").fetchall()
    transit_conn.execute("DROP TABLE fare_rules")
    transit_conn.execute(FARE_RULES_SQL)
    transit_conn.executemany("INSERT INTO fare_rules VALUES (?, ?, ?, ?, ?)", rows)
