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
    """Migration to add turn restrictions support and allow_uturns setting."""
    logger.info("Beginning migration to add turn restrictions support")

    # Add allow_uturns to the about table if it doesn't exist
    cursor = project_conn.execute("SELECT 1 FROM about WHERE infoname = 'allow_uturns'")
    if cursor.fetchone() is None:
        project_conn.execute("INSERT INTO about (infoname, infovalue) VALUES ('allow_uturns', '0')")
        logger.info("Added 'allow_uturns' setting to about table")

    # Create turn_restrictions table (this migration assumes it did not previously exist)
    schema = pathlib.Path(__file__).parent.parent / "tables" / "turn_restrictions.sql"
    if schema.exists():
        run_queries_from_sql_file(project_conn, schema)
        logger.info("Created turn_restrictions table")

        trigger_sql = pathlib.Path(__file__).parent.parent / "triggers" / "turn_restrictions_triggers.sql"
        if trigger_sql.exists():
            run_queries_from_sql_file(project_conn, trigger_sql)
            logger.info("Applied turn restriction triggers")
    else:
        logger.warning(f"Could not find turn_restrictions.sql at {schema}")

    project_conn.commit()
    logger.info("Migration for turn restrictions support completed")
