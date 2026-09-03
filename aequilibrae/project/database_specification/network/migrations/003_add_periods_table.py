import pathlib
import sqlite3
from typing import Optional

from aequilibrae.log import logger
from aequilibrae.project.project_creation import run_queries_from_sql_file


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    logger.info("Beginning migration to add period table to the main project_database.sqlite")
    if project_conn.execute("PRAGMA table_info(periods)").fetchone() is None:
        logger.info("Table does not exist, adding")
        schema = pathlib.Path(__file__).parent.parent / "tables" / "periods.sql"
        run_queries_from_sql_file(project_conn, logger, schema)
    else:
        logger.info("Table already exists")
