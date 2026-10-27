import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# The fare_rules definition this migration creates, frozen here so later changes to tables/fare_rules.sql do not
# change what this migration does
FARE_RULES_SQL = """
create TABLE IF NOT EXISTS fare_rules (
	fare_id     INTEGER  NOT NULL,
	route_id    INTEGER,
	origin      TEXT,
	destination TEXT,
	contains    TEXT,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred
);
"""

COLUMNS = ["fare_id", "route_id", "origin", "destination", "contains"]


def _as_text(value):
    return None if value is None else str(value)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Store the zones of fare rules as text, as GTFS zone IDs are text.

    ``fare_rules.contains`` was declared ``INTEGER``, so zone IDs that look like numbers (such as ``07``) were
    converted. Values converted before this migration cannot be recovered and are kept as their text form.
    """
    if transit_conn is None:
        raise RuntimeError("Transit migration 008 requires a transit_conn connection")

    tables = {row[0] for row in transit_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "fare_rules" not in tables:
        logger.info("Migration finished, transit schema does not contain fare rules.")
        return

    declared = {row[1]: row[2].upper() for row in transit_conn.execute("PRAGMA table_info(fare_rules)")}
    if declared.get("contains") == "TEXT":
        logger.info("Migration finished, fare rule zones are already text.")
        return

    columns = ", ".join(COLUMNS)
    rows = transit_conn.execute(f"SELECT {columns} FROM fare_rules").fetchall()
    transit_conn.execute("DROP TABLE fare_rules")
    transit_conn.execute(FARE_RULES_SQL)

    rows = [(fare_id, route_id, *map(_as_text, zones)) for fare_id, route_id, *zones in rows]
    transit_conn.executemany(f"INSERT INTO fare_rules ({columns}) VALUES (?, ?, ?, ?, ?)", rows)
    logger.info(f"Stored the zones of {len(rows)} fare rules as text")
