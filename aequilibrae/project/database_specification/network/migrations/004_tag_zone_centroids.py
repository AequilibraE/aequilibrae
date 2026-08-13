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
    logger.info("Beginning migration to enforce the centroid flag on nodes that share their ID with a zone")

    sql = "SELECT count(*) FROM sqlite_master WHERE type='table' AND lower(name)='zones'"
    if project_conn.execute(sql).fetchone()[0] == 0:
        logger.info("Migration finished, no 'zones' table found.")
        return

    # Nodes that already share their ID with a zone are tagged before the triggers that enforce it are added
    sql = "UPDATE nodes SET is_centroid=1 WHERE is_centroid != 1 AND node_id IN (SELECT zone_id FROM zones)"
    tagged = project_conn.execute(sql).rowcount
    logger.info(f"{tagged} nodes were tagged as centroids")

    triggers = pathlib.Path(__file__).parent.parent / "triggers" / "zones_triggers.sql"
    run_queries_from_sql_file(project_conn, logger, triggers)
