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
        raise RuntimeError("Network migration 007 requires a project_conn connection")

    if project_conn.execute("PRAGMA table_info(zones)").fetchone() is not None:
        logger.info("Zones table already exists. Nothing was done.")
        return

    schema = pathlib.Path(__file__).parent.parent / "tables" / "zones.sql"
    run_queries_from_sql_file(project_conn, schema)
