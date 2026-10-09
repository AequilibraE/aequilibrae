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
    src_exists = _table_exists(project_conn, SRC_TABLE)

    if src_exists and project_conn.execute(f'SELECT COUNT(*) FROM main."{SRC_TABLE}"').fetchone()[0] == 0:
        # it has a table, but no results yet. We just drop the empty table from the project database
        # carefully drop
        try:
            # project_conn.execute("BEGIN")
            project_conn.execute(f'DROP TABLE "{SRC_TABLE}"')
            # project_conn.commit()
        except Exception:
            project_conn.rollback()
            raise
        return

    if results_conn is None:
        raise (ValueError("result_conn is required, but is not given"))
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

    assert set(cols.split(", ")) == set(existing_columns), ValueError(
        f"project database has different columns. Expected: {set(cols.split(', '))}, got {set(existing_columns)}"
    )

    project_conn.execute("ATTACH DATABASE ? AS results_db", (results_path,))
    attached = {name: path for _seq, name, path in project_conn.execute("PRAGMA database_list")}
    if attached.get("results_db") != results_path:
        raise RuntimeError("results_db is not attached to the expected file")

    add_blank_results_table(results_conn)

    try:
        # project_conn.execute("BEGIN")

        src_count = project_conn.execute(f'SELECT COUNT(*) FROM main."{SRC_TABLE}"').fetchone()[0]

        # copy the table
        project_conn.execute(f'INSERT INTO results_db."{DST_TABLE}" ({cols}) SELECT {cols} FROM main."{SRC_TABLE}"')

        dst_count = project_conn.execute(f'SELECT COUNT(*) FROM results_db."{DST_TABLE}"').fetchone()[0]
        if src_count != dst_count:
            raise RuntimeError(f"Row count mismatch: source={src_count}, destination={dst_count}")

        # project_conn.commit()

    except Exception:
        project_conn.rollback()
        logger.exception("Copy failed. Rollback project ")
        raise
    finally:
        # detach
        project_conn.execute("DETACH DATABASE results_db")

    # carefully drop
    try:
        project_conn.execute("BEGIN")
        project_conn.execute(f'DROP TABLE "{SRC_TABLE}"')
        project_conn.commit()
    except Exception:
        project_conn.rollback()
        raise


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
