import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    if project_conn is None:
        raise RuntimeError("Transit migration 004 requires a project_conn connection")
    if transit_conn is None:
        raise RuntimeError("Transit migration 004 requires a transit_conn connection")

    logger.info("Beginning migration to move transit results to the main project database")

    if transit_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'").fetchone() is None:
        logger.info("Migration finished, no table 'results' in 'public_transport.sqlite'.")
        return
    if results_conn is None:
        raise RuntimeError("Transit migration 004 requires a results_conn connection when transit results exist")

    payloads = {
        row[0]
        for row in results_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    recorded = {row[0] for row in project_conn.execute("SELECT table_name FROM results").fetchall()}
    for table_name in payloads - recorded:
        project_conn.execute(
            "INSERT INTO results (table_name, procedure, procedure_id, procedure_report) VALUES (?, '', '', 'null')",
            (table_name,),
        )

    project_columns = {row[1] for row in project_conn.execute("PRAGMA table_info(results)").fetchall()}
    transit_columns = [row[1] for row in transit_conn.execute("PRAGMA table_info(results)").fetchall()]
    columns = [column for column in transit_columns if column in project_columns]
    if "table_name" in columns:
        quoted = ",".join(f'"{column}"' for column in columns)
        updates = ",".join(f'"{column}"=excluded."{column}"' for column in columns if column != "table_name")
        sql = f"INSERT INTO results ({quoted}) VALUES ({','.join('?' for _ in columns)})"
        if updates:
            sql += f' ON CONFLICT("table_name") DO UPDATE SET {updates}'
        rows = transit_conn.execute(f"SELECT {quoted} FROM results").fetchall()
        project_conn.executemany(sql, rows)

    transit_conn.execute("DROP TABLE results")
