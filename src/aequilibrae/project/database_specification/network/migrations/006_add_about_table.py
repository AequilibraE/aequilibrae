import logging
import pathlib
import sqlite3
import uuid
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
        raise RuntimeError("Network migration 006 requires a project_conn connection")

    if project_conn.execute("PRAGMA table_info(about)").fetchone() is not None:
        logger.info("About table already exists. Nothing was done.")
        return

    schema = pathlib.Path(__file__).parent.parent / "tables" / "about.sql"
    run_queries_from_sql_file(project_conn, schema)
    project_conn.execute("UPDATE about SET infovalue=? WHERE infoname='project_id'", (uuid.uuid4().hex,))
    project_conn.execute("UPDATE about SET infovalue='right' WHERE infoname='driving_side'")
