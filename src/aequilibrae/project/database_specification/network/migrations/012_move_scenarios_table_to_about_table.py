import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT_NAME = "root"
ROOT_DESCRIPTION = (
    'The default, and root, scenario for an AequilibraE project. The name "root" is treated as a special case.'
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Returns if 'conn' has a table named 'name'"""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def _set_about(conn: sqlite3.Connection, infoname: str, infovalue: Optional[str]) -> None:
    """Write one record of the about table, adding the record if it is missing."""
    if conn.execute("UPDATE about SET infovalue=? WHERE infoname=?", (infovalue, infoname)).rowcount == 0:
        conn.execute("INSERT INTO about (infoname, infovalue) VALUES(?, ?)", (infoname, infovalue))


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    """Move each scenario's name and description from the root's scenarios table into that scenario's about table."""
    if project_conn is None:
        raise RuntimeError("Network migration 012 requires a project_conn connection")

    # Only the root database has a scenarios table, the root's migration updates the about table of other scenarios
    if not _table_exists(project_conn, "scenarios"):
        return

    logger.info("Beginning migration to move the scenarios table into the about table of each scenario")
    scenarios = dict(project_conn.execute("SELECT scenario_name, description FROM scenarios").fetchall())
    root_description = scenarios.pop(ROOT_NAME, ROOT_DESCRIPTION)

    scenarios_folder = None
    if scenarios:
        database_path = next(row[2] for row in project_conn.execute("PRAGMA database_list") if row[1] == "main")
        if not database_path:
            raise ValueError("Scenario migration requires an on-disk project database")
        scenarios_folder = Path(database_path).parent / "scenarios"

    for name, description in scenarios.items():
        path = scenarios_folder / name / "project_database.sqlite"
        if not path.is_file():
            logger.warning("Dropping scenario %r, its project database %r does not exist", name, str(path))
            continue

        # SQLite rollback cannot undo these writes, but they are idempotent if the root's migration fails and reruns.
        # Scenario databases hold a copy of the about table they were cloned from, so both records are replaced.
        conn = sqlite3.connect(path)
        try:
            if not _table_exists(conn, "about"):
                raise RuntimeError(f"Scenario {name!r} has no about table, upgrade it before the root scenario")
            with conn:
                _set_about(conn, "scenario_name", name)
                _set_about(conn, "scenario_description", description)
        finally:
            conn.close()

    _set_about(project_conn, "scenario_name", ROOT_NAME)
    _set_about(project_conn, "scenario_description", root_description)

    project_conn.execute("DROP TABLE scenarios")
    if _table_exists(project_conn, "attributes_documentation"):
        project_conn.execute("DELETE FROM attributes_documentation WHERE name_table='scenarios'")
