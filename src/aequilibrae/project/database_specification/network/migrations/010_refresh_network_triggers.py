import logging
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import add_triggers, remove_triggers

logger = logging.getLogger(__name__)

def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    """Refresh network triggers, preserving existing data, tables and indexes."""
    if project_conn is None:
        raise RuntimeError("Network migration 010 requires a project_conn connection")

    logger.info("Beginning migration to refresh the network triggers")

    # Trigger names are unchanged, so dropping the current names removes every outdated definition
    remove_triggers(project_conn, "network")

    add_triggers(project_conn, "network")
    logger.info("Migration to refresh the network triggers completed")
