import logging
import sqlite3
from pathlib import Path
from typing import Optional

from aequilibrae.project.network.links_view import LINK_ATTRIBUTES_TABLE, MANDATORY_LINK_FIELDS, refresh_links_view
from aequilibrae.project.project_creation import _add_triggers, _remove_triggers
from aequilibrae.utils.db_utils import escape_identifier

logger = logging.getLogger(__name__)

_LINKS_SQL = Path(__file__).resolve().parent.parent / "tables" / "links.sql"


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    """Split links into links (geometry and mandatory fields) and link_attributes (all other fields).

    Every non-mandatory column of links, including user-defined ones, moves to link_attributes with its
    declared type and default, and the two tables are presented together by the editable links_full view.
    Indexes covering only attribute columns move along with them. Descriptions in attributes_documentation
    stay under the links table name. No field is added or removed: projects keep exactly the link fields they
    had, whether or not new projects get them.

    links keeps its own definition, constraints included, minus the attribute columns. The exception are
    tables created from the specification that preceded the split, which attaches its CHECK constraints to
    the other_attributes column: dropping that column would silently drop them, so those tables are rebuilt
    from the current specification instead (which they already match), keeping each link's ogc_fid.
    """
    if project_conn is None:
        raise RuntimeError("Network migration 012 requires a project_conn connection")
    conn = project_conn

    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [LINK_ATTRIBUTES_TABLE]).fetchone():
        logger.info("Links are already split into links and link_attributes")
        refresh_links_view(conn)
        return

    logger.info("Beginning migration to split links into links and link_attributes")
    columns = conn.execute("PRAGMA table_info(links)").fetchall()
    attributes = [column for column in columns if column[1].lower() not in MANDATORY_LINK_FIELDS]
    attribute_names = {column[1].lower() for column in attributes}

    _warn_about_dependent_views(conn)
    moving_indexes = _drop_attribute_indexes(conn, attribute_names)

    # 1. Move the attributes, keeping declared types and defaults
    definitions = ["link_id INTEGER PRIMARY KEY"]
    for _, name, data_type, _, default, _ in attributes:
        definition = f"{escape_identifier(name)} {data_type or ''}".rstrip()
        definitions.append(definition if default is None else f"{definition} DEFAULT {default}")
    conn.execute(f"CREATE TABLE {LINK_ATTRIBUTES_TABLE} ({', '.join(definitions)})")

    quoted = ", ".join(["link_id"] + [escape_identifier(column[1]) for column in attributes])
    conn.execute(f"INSERT INTO {LINK_ATTRIBUTES_TABLE} ({quoted}) SELECT {quoted} FROM links")

    # 2. Remove them from links
    _remove_triggers(conn, "network")
    links_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='links'").fetchone()[0]
    if "CHECK" in links_sql.upper():
        _rebuild_links(conn)
    else:
        for column in attributes:
            conn.execute(f"ALTER TABLE links DROP COLUMN {escape_identifier(column[1])}")

    # 3. Indexes, triggers and the links_full view
    for index_name, index_columns in moving_indexes:
        quoted_columns = ", ".join(escape_identifier(column) for column in index_columns)
        conn.execute(f"CREATE INDEX {escape_identifier(index_name)} ON {LINK_ATTRIBUTES_TABLE} ({quoted_columns})")

    _add_triggers(conn, "network")
    refresh_links_view(conn)
    logger.info(f"Moved {len(attributes)} link fields to link_attributes")


def _rebuild_links(conn: sqlite3.Connection) -> None:
    """Recreate links from the specification with its mandatory fields, keeping every link's ogc_fid."""
    fields = ", ".join(MANDATORY_LINK_FIELDS)
    conn.execute(f"CREATE TEMP TABLE _links_before_split AS SELECT {fields} FROM links")

    # Dropping the table also drops its triggers. Its SpatiaLite registration and index go first.
    conn.execute("SELECT DisableSpatialIndex('links', 'geometry')").fetchone()
    conn.execute("DROP TABLE IF EXISTS idx_links_geometry")
    conn.execute("SELECT DiscardGeometryColumn('links', 'geometry')").fetchone()
    conn.execute("DROP TABLE links")

    for statement in _LINKS_SQL.read_text().split("--#"):
        code = "\n".join(line for line in statement.splitlines() if not line.lstrip().startswith("--"))
        if not code.strip().upper().startswith("INSERT"):  # Field descriptions are kept as they are
            conn.execute(statement)

    conn.execute(f"INSERT INTO links ({fields}) SELECT {fields} FROM temp._links_before_split")
    conn.execute("DROP TABLE temp._links_before_split")


def _drop_attribute_indexes(conn: sqlite3.Connection, attribute_columns: set[str]) -> list[tuple[str, list[str]]]:
    """Drop the indexes on links that cover attribute columns, returning those to recreate on link_attributes.

    SQLite cannot drop an indexed column. Indexes covering only attribute columns are recreated on
    link_attributes, while indexes mixing them with mandatory fields cannot be, and are only dropped.
    """
    moving = []
    sql = "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='links' AND sql IS NOT NULL"
    for (index_name,) in conn.execute(sql).fetchall():
        index_columns = [row[2] for row in conn.execute(f"PRAGMA index_info({escape_identifier(index_name)})")]
        lowered = {column.lower() for column in index_columns if column}
        if not lowered & attribute_columns:
            continue
        if lowered <= attribute_columns:
            moving.append((index_name, index_columns))
        else:
            logger.warning(f"Index {index_name} mixes mandatory and attribute link fields and is dropped")
        conn.execute(f"DROP INDEX {escape_identifier(index_name)}")
    return moving


def _warn_about_dependent_views(conn: sqlite3.Connection) -> None:
    sql = "SELECT name FROM sqlite_master WHERE type='view' AND name != 'links_full' AND sql LIKE '%links%'"
    views = [row[0] for row in conn.execute(sql).fetchall()]
    if views:
        logger.warning(f"These views may reference link fields that moved to link_attributes: {views}")
