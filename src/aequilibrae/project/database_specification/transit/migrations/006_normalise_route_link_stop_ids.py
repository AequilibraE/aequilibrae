import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def _normalise_stop_column(conn: sqlite3.Connection, column: str) -> None:
    """Replace legacy external GTFS stop codes with internal ``stop_id`` values."""
    unresolved = conn.execute(
        f"""
        SELECT DISTINCT route_links."{column}"
        FROM route_links
        WHERE NOT EXISTS (
            SELECT 1 FROM stops WHERE stops.stop_id = CAST(route_links."{column}" AS TEXT)
        )
        AND (SELECT COUNT(*) FROM stops WHERE stops.stop = CAST(route_links."{column}" AS TEXT)) != 1
        """
    ).fetchall()
    if unresolved:
        values = [row[0] for row in unresolved[:10]]
        raise ValueError(f"cannot normalise route_links.{column}: no unique matching stop for {values}")

    conn.execute(
        f"""
        UPDATE route_links
        SET "{column}" = (
            SELECT stops.stop_id FROM stops WHERE stops.stop = CAST(route_links."{column}" AS TEXT)
        )
        WHERE NOT EXISTS (
            SELECT 1 FROM stops WHERE stops.stop_id = CAST(route_links."{column}" AS TEXT)
        )
        """
    )


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Normalise legacy route-link stop references to ``stops.stop_id``."""
    if transit_conn is None:
        raise RuntimeError("Transit migration 006 requires a transit_conn connection")

    tables = {row[0] for row in transit_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"route_links", "stops"} <= tables:
        logger.info("Migration finished, transit schema does not contain route links and stops.")
        return

    _normalise_stop_column(transit_conn, "from_stop")
    _normalise_stop_column(transit_conn, "to_stop")

    violations = transit_conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError(f"transit foreign-key check failed: {violations[:10]}")

    logger.info("Normalised route-link stop identifiers")
