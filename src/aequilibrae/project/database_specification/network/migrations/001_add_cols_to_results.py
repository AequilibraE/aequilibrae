import sqlite3
from typing import Optional


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 001 requires a project_conn connection")

    project_conn.execute("ALTER TABLE results ADD COLUMN year TEXT")
    project_conn.execute("ALTER TABLE results ADD COLUMN scenario TEXT")
    project_conn.execute("ALTER TABLE results ADD COLUMN reference_table TEXT")
