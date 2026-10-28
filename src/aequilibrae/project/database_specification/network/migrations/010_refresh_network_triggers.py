import logging
import re
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import add_triggers, remove_triggers

logger = logging.getLogger(__name__)

# Single-column indexes the network triggers rely on, named as in the current table specifications
REQUIRED_INDEXES = (
    ("idx_link_anode", "links", "a_node"),
    ("idx_link_bnode", "links", "b_node"),
    ("idx_link_link_type", "links", "link_type"),
)

# Indexes created by earlier specifications that duplicate another index on the same columns
REDUNDANT_INDEXES = {
    "links": (
        "aequilibrae_links_a_node_idx",
        "links_a_node_idx",
        "aequilibrae_links_b_node_idx",
        "links_b_node_idx",
        "aequilibrae_links_link_type",
        "links_link_type",
        "idx_link",
    ),
    "nodes": ("aequilibrae_nodes_node_id", "nodes_node_id", "idx_node"),
}

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
    if project_conn is None:
        raise RuntimeError("Network migration 010 requires a project_conn connection")

    logger.info("Beginning migration to refresh the network triggers")

    # Trigger names are unchanged, so dropping the current names removes every outdated definition
    remove_triggers(project_conn, "network")

    refreshed = _refresh_node_attributes(project_conn)
    if refreshed:
        logger.warning("Refreshed the modes and link types of %s nodes", refreshed)

    _drop_redundant_indexes(project_conn)
    _allow_empty_link_endpoints(project_conn)

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


def _drop_redundant_indexes(conn: sqlite3.Connection):
    for name, table, column in REQUIRED_INDEXES:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})")

    for table, names in REDUNDANT_INDEXES.items():
        indexes = {}
        for _, name, unique, _, partial in conn.execute(f"PRAGMA index_list({table})").fetchall():
            if not partial:
                columns = tuple(row[2] for row in conn.execute(f'PRAGMA index_info("{name}")'))
                indexes[name] = (columns, unique)

        # Only drop an index when another one with the same columns, and at least as strict, remains
        for name in names:
            if name not in indexes:
                continue
            columns, unique = indexes.pop(name)
            if any(cols == columns and uniq >= unique for cols, uniq in indexes.values()):
                conn.execute(f'DROP INDEX "{name}"')
                logger.info("Dropped redundant index %s", name)
            else:
                indexes[name] = (columns, unique)


def _allow_empty_link_endpoints(conn: sqlite3.Connection):
    """Lets links be inserted without a_node/b_node, which the triggers fill in from the geometry"""
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'links'").fetchone()[0]
    new_sql = sql
    for field in ("a_node", "b_node"):
        pattern = rf"CHECK\s*\(\s*TYPEOF\s*\(\s*{field}\s*\)\s*==?\s*'integer'\s*\)"
        new_sql = re.sub(pattern, f"CHECK(TYPEOF({field}) IN ('integer', 'null'))", new_sql, flags=re.IGNORECASE)
    if new_sql == sql:
        return

    # SQLite cannot alter a CHECK constraint in place, but loosening one leaves the stored rows valid, so the
    # writable_schema procedure documented at https://www.sqlite.org/lang_altertable.html#otheralter applies
    version = conn.execute("PRAGMA schema_version").fetchone()[0]
    conn.execute("PRAGMA writable_schema = ON")
    try:
        conn.execute("UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = 'links'", (new_sql,))
        conn.execute(f"PRAGMA schema_version = {version + 1}")
    except sqlite3.DatabaseError as e:
        # SQLite builds running in defensive mode refuse schema edits. Inserts then keep needing a placeholder
        logger.warning("Could not allow empty a_node/b_node on links: %s", e)
    finally:
        conn.execute("PRAGMA writable_schema = OFF")
