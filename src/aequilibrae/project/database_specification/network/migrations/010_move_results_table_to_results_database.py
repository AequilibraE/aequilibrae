import logging
import sqlite3
from typing import Optional


logger = logging.getLogger(__name__)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if results_conn is None:
        raise (ValueError("connect to results table is not given"))
    logger.info("Beginning migration to move the results table in project_database.sqlite to results_database.sqlite")

    # check/ reserve "summary"
    # name = "summary"
    # try:
    #     with sqlite3.connect("app.db") as conn:
    #         cursor = conn.cursor()
    #         cursor.execute(
    #             "INSERT INTO names_registry (name) VALUES (?)", (name,)
    #         )
    #         conn.commit()
    #         print(f"Name '{name}' successfully reserved!")
    # except sqlite3.IntegrityError:
    #     print(f"Name '{name}' is already taken.")
    results_path = results_conn.execute("PRAGMA database_list").fetchone()[2]
    project_conn.execute("ATTACH DATABASE ? AS results_db", (results_path,))

    SRC_TABLE = "results"  # name in project_database.sqlite
    DST_TABLE = "summary"  # name in results_database.sqlite

    results_conn.execute(f"""
        CREATE TABLE {DST_TABLE} (
            table_name       TEXT     NOT NULL PRIMARY KEY,
            procedure        TEXT     NOT NULL,
            procedure_id     TEXT     NOT NULL UNIQUE,
            procedure_report TEXT     NOT NULL,
            timestamp        DATETIME DEFAULT current_timestamp,
            description      TEXT, year TEXT, scenario TEXT, reference_table TEXT
        )
    """)
    results_conn.commit()

    cols = (
        "table_name, procedure, procedure_id, procedure_report, timestamp, description, year, scenario, reference_table"
    )
    project_conn.execute(f'INSERT INTO results_db."{DST_TABLE}" ({cols}) SELECT {cols} FROM main."{SRC_TABLE}"')

    project_conn.execute(f"DROP TABLE {SRC_TABLE};")
    # have copied - do some check

    # check it doesn't already exist

    # make a new summary

    # read in data

    # if works, close

    # be very careful


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
