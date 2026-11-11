import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PERIOD_END = 172800  # 48:00:00, the latest time of a GTFS service day


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    """Extend the default period to the whole service day, so it keeps trips that run past midnight."""
    if project_conn is None:
        raise RuntimeError("Network migration 012 requires a project_conn connection")

    if project_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='periods'").fetchone() is None:
        logger.info("Migration finished, no periods table found.")
        return

    # The trigger rejects any change to the default period, so it is recreated around the update
    project_conn.execute("DROP TRIGGER IF EXISTS aequilibrae_default_period_update")
    project_conn.execute(
        "UPDATE periods SET period_start = 0, period_end = ?, "
        "period_description = 'Default time period, whole service day' WHERE period_id = 1",
        [DEFAULT_PERIOD_END],
    )
    project_conn.execute(
        """CREATE TRIGGER aequilibrae_default_period_update BEFORE UPDATE ON periods
        WHEN new.period_id = 1
        BEGIN
            SELECT RAISE(ABORT, 'Cannot update default period');
        END"""
    )
    logger.info("The default period now spans the whole service day, 00:00:00 to 48:00:00")
