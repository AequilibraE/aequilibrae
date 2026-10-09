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
    logger.info("Beginning migration to move the results table in project_database.sqlite to results_database.sqlite")
