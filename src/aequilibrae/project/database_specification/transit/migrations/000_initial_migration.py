import sqlite3
from typing import Optional


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if transit_conn is None:
        raise RuntimeError("Transit migration 000 requires a transit_conn connection")

    transit_conn.execute(
        """CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY CHECK( id >= 0),
            name TEXT NOT NULL,
            status TEXT DEFAULT 'MISSING' CHECK( status IN ('APPLIED', 'SKIPPED', 'MISSING') ) NOT NULL,
            date TIMESTAMP DEFAULT NULL
        )"""
    )
