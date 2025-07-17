import sqlite3
from typing import Optional
import pathlib

from aequilibrae import Project, logger
from aequilibrae.context import get_active_project

from aequilibrae.project.project_creation import run_queries_from_sql_file


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    logger.info("Beginning migration to add scenario support to the main project_database.sqlite")
    schema = pathlib.Path(__file__).parent.parent / "tables" / "scenarios.sql"
    run_queries_from_sql_file(project_conn, logger, schema)
