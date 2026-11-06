import logging
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import add_triggers, initialize_tables


logger = logging.getLogger(__name__)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Returns if 'conn' has a table named 'name'"""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 010 requires a project_conn connection")
    src_exists = _table_exists(project_conn, "results")
    dst_exists = results_conn is not None and _table_exists(results_conn, "results")
    if dst_exists:
        add_triggers(results_conn, "results")

    if src_exists and project_conn.execute("SELECT COUNT(*) FROM main.results").fetchone()[0] == 0:
        # It has a table, but no results yet. Drop it from the project database and initialise
        # the results database if one is already present.
        if results_conn is not None and not dst_exists:
            initialize_tables("results", results_conn)
        project_conn.execute("DROP TABLE results")
        return

    if results_conn is None:
        if not src_exists:
            logger.info(
                "the results table is not in the project database, and "
                "results connection not given, nothing to migrate."
            )
            return
        num_rows = project_conn.execute("SELECT COUNT(*) FROM main.results").fetchall()[0]
        raise (
            ValueError(
                f"There are {num_rows} rows in the results table in the project database, but result_conn is not given"
            )
        )
    logger.info("Beginning migration to move the results table in project_database.sqlite to results_database.sqlite")

    # Check if it already exists.
    if not src_exists and dst_exists:
        logger.info("Results summary table is already in the results database, and not in the project database.")
        return
    if not src_exists and not dst_exists:
        initialize_tables("results", results_conn)
        return
    if src_exists and dst_exists:
        # Possibly a previous run crashed after copying but before dropping.
        raise RuntimeError(
            "results table already exists in the results database, but it still exists in the project database."
        )

    cols = [
        "table_name",
        "procedure",
        "procedure_id",
        "procedure_report",
        "timestamp",
        "description",
        "year",
        "scenario",
        "reference_table",
    ]
    col_names = ", ".join(cols)
    existing_columns = [row[1] for row in project_conn.execute('PRAGMA table_info("results")')]

    if set(cols) != set(existing_columns):
        raise ValueError(f"project database has different columns. Expected: {set(cols)}, got {set(existing_columns)}")
    initialize_tables("results", results_conn)

    data = project_conn.execute(f"SELECT {col_names} FROM results").fetchall()

    question_marks = ", ".join(["?" for _ in cols])
    results_conn.executemany(f"INSERT INTO results({col_names}) VALUES({question_marks})", data)

    project_conn.execute("DROP TABLE results")
