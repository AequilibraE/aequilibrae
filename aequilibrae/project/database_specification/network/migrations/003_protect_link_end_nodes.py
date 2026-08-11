import sqlite3
from typing import Optional

from aequilibrae.log import logger
from aequilibrae.project.project_creation import add_triggers, remove_triggers


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    logger.info("Beginning migration to protect links.a_node and links.b_node")

    if project_conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone()[0] != 1:
        logger.warning("Recovering the nodes spatial index before validating link endpoints")
        recovered = project_conn.execute("SELECT RecoverSpatialIndex('nodes', 'geometry')").fetchone()[0]
        valid = project_conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone()[0]
        if recovered != 1 or valid != 1:
            raise RuntimeError("Cannot protect link endpoint fields because the nodes spatial index is invalid")

    # Remove both the current prefixed names and their legacy unprefixed forms
    # before installing the current trigger set. Retired topology names are no
    # longer discoverable from the specification and must be removed explicitly.
    remove_triggers(project_conn, logger, "network", use_aequilibrae_prefix=True)
    remove_triggers(project_conn, logger, "network", use_aequilibrae_prefix=False)

    retired_triggers = (
        "aequilibrae_cannibalize_node",
        "cannibalize_node",
        "aequilibrae_cannibalise_node",
        "cannibalise_node",
        "aequilibrae_cannibalise_node_abort_when_centroid",
        "cannibalise_node_abort_when_centroid",
        "updated_link_geometry",
    )
    for trigger in retired_triggers:
        project_conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    add_triggers(project_conn, logger, "network")

    repaired = 0
    for field, endpoint_function in (("a_node", "StartPoint"), ("b_node", "EndPoint")):
        cursor = project_conn.execute(
            f"""
            UPDATE links
            SET {field} = (
                SELECT node_id
                FROM nodes
                WHERE nodes.geometry = {endpoint_function}(links.geometry)
                  AND nodes.ROWID IN (
                      SELECT ROWID
                      FROM SpatialIndex
                      WHERE f_table_name = 'nodes'
                        AND search_frame = {endpoint_function}(links.geometry)))
            WHERE NOT EXISTS (
                SELECT 1
                FROM nodes
                WHERE nodes.node_id = links.{field}
                  AND nodes.geometry = {endpoint_function}(links.geometry))
              AND 1 = (
                SELECT count(*)
                FROM nodes
                WHERE nodes.geometry = {endpoint_function}(links.geometry)
                  AND nodes.ROWID IN (
                      SELECT ROWID
                      FROM SpatialIndex
                      WHERE f_table_name = 'nodes'
                        AND search_frame = {endpoint_function}(links.geometry)))
            """
        )
        repaired += cursor.rowcount

    unresolved_links = project_conn.execute(
        """
        SELECT link_id
        FROM links
        WHERE NOT EXISTS (
            SELECT 1
            FROM nodes
            WHERE nodes.node_id = links.a_node
              AND nodes.geometry = StartPoint(links.geometry))
           OR NOT EXISTS (
            SELECT 1
            FROM nodes
            WHERE nodes.node_id = links.b_node
              AND nodes.geometry = EndPoint(links.geometry))
        ORDER BY link_id
        LIMIT 20
        """
    ).fetchall()
    if unresolved_links:
        link_ids = [row[0] for row in unresolved_links]
        raise RuntimeError(
            "Cannot protect link endpoint fields because some endpoints have no unique matching node. "
            f"First affected link IDs: {link_ids}"
        )

    if repaired:
        logger.warning("Repaired %s inconsistent link endpoint assignments", repaired)

    logger.info("Migration to protect links.a_node and links.b_node completed")
