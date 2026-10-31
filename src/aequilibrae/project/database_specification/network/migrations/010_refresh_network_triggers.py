import logging
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import add_triggers, remove_triggers

logger = logging.getLogger(__name__)

NODE_ATTRIBUTES = """
    CREATE TEMP TABLE aequilibrae_node_attributes AS
    SELECT node_id,
           (SELECT GROUP_CONCAT(mode_id, '')
            FROM modes
            WHERE instr((SELECT GROUP_CONCAT(modes, '')
                         FROM links
                         WHERE links.a_node = nodes.node_id OR links.b_node = nodes.node_id), mode_id) > 0) AS modes,
           (SELECT GROUP_CONCAT(link_type_id, '')
            FROM link_types
            WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '')
                         FROM links
                         INNER JOIN link_types ON links.link_type = link_types.link_type
                         WHERE links.a_node = nodes.node_id OR links.b_node = nodes.node_id), link_type_id) > 0)
             AS link_types
    FROM nodes
"""


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    """Refresh network triggers and derived node attributes, preserving tables and indexes."""
    if project_conn is None:
        raise RuntimeError("Network migration 010 requires a project_conn connection")

    logger.info("Beginning migration to refresh the network triggers")

    # Trigger names are unchanged, so dropping the current names removes every outdated definition
    remove_triggers(project_conn, "network")

    refreshed = _refresh_node_attributes(project_conn)
    if refreshed:
        logger.warning("Refreshed the modes and link types of %s nodes", refreshed)

    add_triggers(project_conn, "network")
    logger.info("Migration to refresh the network triggers completed")


def _refresh_node_attributes(conn: sqlite3.Connection) -> int:
    """Recomputes nodes.modes and nodes.link_types, which link deletions used to leave stale"""
    conn.execute("DROP TABLE IF EXISTS temp.aequilibrae_node_attributes")
    conn.execute(NODE_ATTRIBUTES)
    conn.execute("CREATE UNIQUE INDEX temp.aequilibrae_node_attributes_id ON aequilibrae_node_attributes (node_id)")
    cursor = conn.execute(
        """
        UPDATE nodes
        SET modes = (SELECT modes FROM aequilibrae_node_attributes AS fresh WHERE fresh.node_id = nodes.node_id),
            link_types = (SELECT link_types
                          FROM aequilibrae_node_attributes AS fresh
                          WHERE fresh.node_id = nodes.node_id)
        WHERE node_id IN (SELECT fresh.node_id
                          FROM aequilibrae_node_attributes AS fresh
                          INNER JOIN nodes AS stale ON stale.node_id = fresh.node_id
                          WHERE stale.modes IS NOT fresh.modes OR stale.link_types IS NOT fresh.link_types)
        """
    )
    conn.execute("DROP TABLE temp.aequilibrae_node_attributes")
    return cursor.rowcount

