import logging
import sqlite3
from pathlib import Path
from typing import Optional

from aequilibrae.project.project_creation import run_queries_from_sql_file

logger = logging.getLogger(__name__)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})") if row[1] != "ogc_fid"]


def _rebuild_child_table(conn: sqlite3.Connection, table: str, tables_path: Path) -> None:
    """Recreate a leaf table after retaining its rows outside the schema."""
    columns = _columns(conn, table)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    rows = conn.execute(f'SELECT {quoted_columns} FROM "{table}"').fetchall()

    conn.execute(f"SELECT DropTable(NULL, '{table}')")
    run_queries_from_sql_file(conn, tables_path / f"{table}.sql")
    if rows:
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})', rows)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Repair invalid transit foreign keys without relaxing FK enforcement."""
    if transit_conn is None:
        logger.info("Migration finished, no 'public_transport.sqlite' connection provided.")
        return

    existing_tables = {
        row[0] for row in transit_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    required_tables = {"stops", "links", "fare_rules", "pattern_mapping"}
    if not required_tables <= existing_tables:
        logger.info("Migration finished, transit schema does not contain the affected tables.")
        return

    stop_columns = _columns(transit_conn, "stops")
    for column, declared_type in (("zone_id", "INTEGER"), ("transit_fare_zone", "TEXT")):
        if column not in stop_columns:
            transit_conn.execute(f"ALTER TABLE stops ADD COLUMN {column} {declared_type}")

    tables_path = Path(__file__).parent.parent / "tables"
    _rebuild_child_table(transit_conn, "links", tables_path)
    _rebuild_child_table(transit_conn, "fare_rules", tables_path)
    _rebuild_child_table(transit_conn, "pattern_mapping", tables_path)

    violations = transit_conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise ValueError(f"transit foreign-key check failed: {violations[:10]}")

    logger.info("Repaired transit foreign-key definitions")
