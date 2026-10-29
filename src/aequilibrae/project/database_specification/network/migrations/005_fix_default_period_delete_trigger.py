import sqlite3
from typing import Optional


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 004 requires a project_conn connection")

    project_conn.execute("DROP TRIGGER IF EXISTS aequilibrae_default_period_delete")
    project_conn.execute("DROP TRIGGER IF EXISTS aequilibrae_root_scenario_delete")
    project_conn.execute(
        """CREATE TRIGGER aequilibrae_default_period_delete BEFORE DELETE ON periods
        WHEN old.period_id = 1
        BEGIN
            SELECT RAISE(ABORT, 'Cannot delete default period');
        END"""
    )
    project_conn.execute(
        """CREATE TRIGGER aequilibrae_root_scenario_delete BEFORE DELETE ON scenarios
        WHEN old.scenario_name = 'root'
        BEGIN
            SELECT RAISE(ABORT, 'Cannot delete root scenario');
        END"""
    )
