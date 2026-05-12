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
    logger.info("Beginning migration to add scenario support to the main project_database.sqlite")
    schema = pathlib.Path(__file__).parent.parent / "tables" / "scenarios.sql"
    run_queries_from_sql_file(project_conn, logger, schema)
