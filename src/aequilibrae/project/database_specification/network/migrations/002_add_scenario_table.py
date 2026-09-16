import logging
import pathlib
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import run_queries_from_sql_file

logger = logging.getLogger(__name__)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 002 requires a project_conn connection")

    logger.info("Beginning migration to add scenario support to the project database")
    schema = pathlib.Path(__file__).parent.parent / "tables" / "scenarios.sql"
    run_queries_from_sql_file(project_conn, schema)
