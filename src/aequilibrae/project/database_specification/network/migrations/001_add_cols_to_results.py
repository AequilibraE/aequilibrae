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

    table_info = project_conn.execute("PRAGMA table_info(results)").fetchall()

    has_year = has_scenario = has_reference_table = False
    for _, name, *_ in table_info:
        if name == "year":
            has_year = True
        elif name == "scenario":
            has_scenario = True
        if name == "reference_table":
            has_reference_table = True

    if not has_year:
        project_conn.execute("ALTER TABLE results ADD COLUMN year TEXT")
    if not has_scenario:
        project_conn.execute("ALTER TABLE results ADD COLUMN scenario TEXT")
    if not has_reference_table:
        project_conn.execute("ALTER TABLE results ADD COLUMN reference_table TEXT")
