import logging
import sqlite3
from typing import Optional

from aequilibrae.utils.db_utils import DST_TABLE, add_blank_results_table


logger = logging.getLogger(__name__)

SRC_TABLE = "results"  # name in project_database.sqlite


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
    src_exists = _table_exists(project_conn, SRC_TABLE)

    if src_exists and project_conn.execute(f'SELECT COUNT(*) FROM main."{SRC_TABLE}"').fetchone()[0] == 0:
        # it has a table, but no results yet. We just drop the empty table from the project database
        # carefully drop
        project_conn.execute(f'DROP TABLE "{SRC_TABLE}"')
        return

    if results_conn is None:
        if not src_exists:
            logger.info(
                f"'{SRC_TABLE}' is not in the project database, and results connection not given, Nothing to migrate."
            )
            return
        num_rows = project_conn.execute('SELECT COUNT(*) FROM main."{}"'.format(SRC_TABLE)).fetchall()[0]
        raise (
            ValueError(
                f"There are {num_rows} rows in the results table in the project database, but result_conn is not given"
            )
        )
    logger.info("Beginning migration to move the results table in project_database.sqlite to results_database.sqlite")

    dst_exists = _table_exists(results_conn, DST_TABLE)

    # check if it already exists
    if not src_exists and dst_exists:
        logger.info("Results summary table is already in the results database, and not in the project database.")
        return
    if not src_exists and not dst_exists:
        raise RuntimeError(
            f"'{SRC_TABLE}' is not in the project database, and {DST_TABLE} is not found in the results database. "
            "Nothing to migrate."
        )
    if src_exists and dst_exists:
        # Possibly a previous run crashed after copying but before dropping.
        raise RuntimeError(
            f"'{DST_TABLE}' already exists in the results database, but {SRC_TABLE} still exists in the project "
            "database."
        )

    results_path = results_conn.execute("PRAGMA database_list").fetchone()[2]
    if results_path == "":
        # results file
        raise ValueError("Cannot find filepath of results database")

    cols = (
        "table_name, procedure, procedure_id, procedure_report, timestamp, description, year, scenario, reference_table"
    )
    existing_columns = [row[1] for row in project_conn.execute('PRAGMA table_info("results")')]

    if set(cols.split(", ")) != set(existing_columns):
        raise ValueError(
            f"project database has different columns. Expected: {set(cols.split(', '))}, got {set(existing_columns)}"
        )
    add_blank_results_table(results_conn)

    data = project_conn.execute(f"select {cols} from {SRC_TABLE}").fetchall()

    question_marks = ", ".join(["?" for col_name in cols.split(", ")])
    results_conn.executemany(f"INSERT INTO {SRC_TABLE}({cols}) VALUES({question_marks})", data)

    project_conn.execute(f'DROP TABLE "{SRC_TABLE}"')


def main():
    with sqlite3.connect(
        "/mnt/c/Users/TylerPearn/Documents/chicago_sample_model/results_database.sqlite"
    ) as results_conn:
        with sqlite3.connect(
            "/mnt/c/Users/TylerPearn/Documents/chicago_sample_model/project_database.sqlite"
        ) as project_conn:
            migrate(project_conn=project_conn, results_conn=results_conn)

            # cursor = results_conn.cursor()

            # # You can now execute SQL queries
            # cursor.execute("SELECT sqlite_version();")
            # version = cursor.fetchone()
            # print(f"Connected! SQLite Version: {version[0]}")

            # Always close the connection when completely done with the database
        #     project_conn.close()
        # results_conn.close()


if __name__ == "__main__":
    main()
