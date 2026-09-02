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

    missing_endpoint_nodes = project_conn.execute(
        """
        SELECT link_id
        FROM links
        WHERE NOT EXISTS (
            SELECT 1
            FROM nodes
            WHERE nodes.node_id = links.a_node)
           OR NOT EXISTS (
            SELECT 1
            FROM nodes
            WHERE nodes.node_id = links.b_node)
        ORDER BY link_id
        LIMIT 20
        """
    ).fetchall()
    if missing_endpoint_nodes:
        link_ids = [row[0] for row in missing_endpoint_nodes]
        raise RuntimeError(
            "Cannot protect link endpoint fields because some endpoints reference missing nodes. "
            f"First affected link IDs: {link_ids}"
        )

    # Preserve the network topology: a_node and b_node are authoritative here.
    # Only reshape the link geometry; node records and geometries remain untouched.
    cursor = project_conn.execute(
        """
        UPDATE links
        SET geometry = SetEndPoint(
            SetStartPoint(
                geometry,
                (SELECT geometry FROM nodes WHERE nodes.node_id = links.a_node)),
            (SELECT geometry FROM nodes WHERE nodes.node_id = links.b_node)),
            distance = GeodesicLength(SetEndPoint(
                SetStartPoint(
                    geometry,
                    (SELECT geometry FROM nodes WHERE nodes.node_id = links.a_node)),
                (SELECT geometry FROM nodes WHERE nodes.node_id = links.b_node)))
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
        """
    )
    repaired = cursor.rowcount

    inconsistent_links = project_conn.execute(
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
    if inconsistent_links:
        link_ids = [row[0] for row in inconsistent_links]
        raise RuntimeError(
            "Cannot protect link endpoint fields because some link geometries could not be aligned with their nodes. "
            f"First affected link IDs: {link_ids}"
        )

    add_triggers(project_conn, logger, "network")

    if repaired:
        logger.warning("Repaired the geometry of %s links with inconsistent endpoints", repaired)

    logger.info("Migration to protect links.a_node and links.b_node completed")
