import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# The scenarios table as specified when this migration was written. Migration 012 moves its contents into the about
# table of each scenario and drops it, so this migration no longer relies on a table specification file.
ROOT_DESCRIPTION = (
    'The default, and root, scenario for an AequilbraE project. The name "root" is treated as a special case.'
)
ATTRIBUTES = [
    ("scenarios", "scenario_name", "The scenario folder name."),
    ("scenarios", "description", "Description of the scenario"),
]


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 002 requires a project_conn connection")

    logger.info("Beginning migration to add scenario support to the project database")
    project_conn.execute("CREATE TABLE IF NOT EXISTS scenarios (scenario_name TEXT UNIQUE NOT NULL, description TEXT)")
    project_conn.execute("INSERT INTO scenarios (scenario_name, description) VALUES('root', ?)", (ROOT_DESCRIPTION,))
    project_conn.executemany(
        "INSERT INTO attributes_documentation (name_table, attribute, description) VALUES(?, ?, ?)", ATTRIBUTES
    )
