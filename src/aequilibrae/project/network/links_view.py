"""The editable ``links_full`` view, which joins ``links`` and ``link_attributes``.

``links`` holds the geometry and the mandatory link fields, and ``link_attributes`` holds every
other field, including user-defined ones. ``links_full`` presents both as a single layer that
QGIS and the API can read and edit.

Writes to the view are routed by INSTEAD OF triggers:

* inserts go to both tables, deletes go to ``links`` (whose triggers remove the attributes);
* each ``links`` column has its own UPDATE trigger, so editing an attribute never sets a ``links``
  column, and the expensive geometry and node triggers on ``links`` only fire when their own
  column changes;
* all attribute columns share one UPDATE trigger, as ``link_attributes`` has no costly triggers.

The view and its triggers list every column of both tables, so they are rebuilt with
:func:`refresh_links_view` whenever a link field is added. :func:`links_source` also rebuilds them
when it finds them missing or out of date, e.g. after a column was added outside AequilibraE.
"""

import logging
import re
from sqlite3 import Connection, OperationalError

from aequilibrae.utils.db_utils import escape_identifier

LINKS_TABLE = "links"
LINK_ATTRIBUTES_TABLE = "link_attributes"
LINKS_VIEW = "links_full"

#: Columns that stay in the links table, everything else is a link attribute
MANDATORY_LINK_FIELDS = (
    "ogc_fid",
    "link_id",
    "a_node",
    "b_node",
    "direction",
    "distance",
    "modes",
    "link_type",
    "geometry",
)

_TRIGGER_PREFIX = "aequilibrae_links_full"

logger = logging.getLogger(__name__)


def links_source(conn: Connection) -> str:
    """Return the relation holding every link field: the ``links_full`` view, or ``links`` on old projects.

    Rebuilds the view when ``link_attributes`` exists but the view is missing or does not show
    every column of the two tables.

    :Arguments:
        **conn** (:obj:`sqlite3.Connection`): Connection to the project database.

    :Returns:
        **relation** (:obj:`str`): ``"links_full"``, or ``"links"`` for projects that predate the split.
    """
    if not _relation_exists(conn, LINK_ATTRIBUTES_TABLE, "table"):
        logger.warning(
            "This project keeps all link attributes in the links table, which is deprecated. "
            "Run project.upgrade() to split them into links and link_attributes."
        )
        return LINKS_TABLE

    if not _view_is_current(conn):
        refresh_links_view(conn)
    return LINKS_VIEW


def has_link_attributes(conn: Connection) -> bool:
    """Return whether the project database splits links into ``links`` and ``link_attributes``."""
    return _relation_exists(conn, LINK_ATTRIBUTES_TABLE, "table")


def refresh_links_view(conn: Connection) -> None:
    """(Re)create the ``links_full`` view, its INSTEAD OF triggers and its SpatiaLite registration.

    :Arguments:
        **conn** (:obj:`sqlite3.Connection`): Connection to the project database, which must already
        have the ``links`` and ``link_attributes`` tables.
    """
    base_columns = _table_info(conn, LINKS_TABLE)
    attribute_columns = _table_info(conn, LINK_ATTRIBUTES_TABLE)
    if not base_columns or not attribute_columns:
        raise RuntimeError("Building the links_full view requires the links and link_attributes tables")

    base_names = {column["name"].lower() for column in base_columns}
    clashing = [c["name"] for c in attribute_columns if c["name"].lower() in base_names and c["name"] != "link_id"]
    if clashing:
        logger.warning(f"link_attributes columns also present in links are left out of {LINKS_VIEW}: {clashing}")
    attribute_columns = [c for c in attribute_columns if c["name"].lower() not in base_names]

    conn.execute(f"DROP VIEW IF EXISTS {LINKS_VIEW}")  # Also drops the view's triggers
    conn.execute(_view_sql(base_columns, attribute_columns))
    for sql in _trigger_sql(base_columns, attribute_columns):
        conn.execute(sql)
    _register_spatial_view(conn)


def _view_sql(base_columns: list[dict], attribute_columns: list[dict]) -> str:
    select = [f"l.{_q(c['name'])} AS {_q(c['name'])}" for c in base_columns if c["name"] != "geometry"]
    select += [f"a.{_q(c['name'])} AS {_q(c['name'])}" for c in attribute_columns]
    select.append('l."geometry" AS "geometry"')
    return (
        f"CREATE VIEW {LINKS_VIEW} AS SELECT {', '.join(select)} "
        f"FROM {LINKS_TABLE} AS l LEFT JOIN {LINK_ATTRIBUTES_TABLE} AS a ON a.link_id = l.link_id"
    )


def _trigger_sql(base_columns: list[dict], attribute_columns: list[dict]) -> list[str]:
    attribute_upsert = _attribute_upsert(attribute_columns)
    sqls = []

    # Inserts write the link, then its attributes. Omitted columns are NULL in NEW, so column defaults
    # are re-applied here (e.g. direction is NOT NULL DEFAULT 0).
    base_names = ", ".join(_q(c["name"]) for c in base_columns)
    base_values = ", ".join(_new_value(c) for c in base_columns)
    body = [f"INSERT INTO {LINKS_TABLE} ({base_names}) VALUES ({base_values});"]
    if attribute_upsert:
        body.append(attribute_upsert.format(link_id="NEW.link_id"))
    sqls.append(_trigger("insert", "INSERT", body))

    # The links triggers delete the attributes as well. The explicit delete covers projects without triggers.
    body = [
        f"DELETE FROM {LINKS_TABLE} WHERE ogc_fid = OLD.ogc_fid;",
        f"DELETE FROM {LINK_ATTRIBUTES_TABLE} WHERE link_id = OLD.link_id;",
    ]
    sqls.append(_trigger("delete", "DELETE", body))

    sqls.append(
        _trigger(
            "update_ogc_fid",
            "UPDATE OF ogc_fid",
            ["SELECT RAISE(ABORT, 'ogc_fid cannot be changed');"],
            when="NEW.ogc_fid IS NOT OLD.ogc_fid",
        )
    )

    # One trigger per links column, so each UPDATE only sets the column that changed
    for column in base_columns:
        name = column["name"]
        if name == "ogc_fid":
            continue
        body = [f"UPDATE {LINKS_TABLE} SET {_q(name)} = NEW.{_q(name)} WHERE ogc_fid = OLD.ogc_fid;"]
        sqls.append(_trigger(f"update_{_slug(name)}", f"UPDATE OF {_q(name)}", body))

    if attribute_upsert:
        # The link_id is looked up so the write lands on the right record even if the same statement renumbers it
        link_id = f"(SELECT link_id FROM {LINKS_TABLE} WHERE ogc_fid = OLD.ogc_fid)"
        columns = ", ".join(_q(c["name"]) for c in attribute_columns)
        sqls.append(_trigger("update_attributes", f"UPDATE OF {columns}", [attribute_upsert.format(link_id=link_id)]))

    return sqls


def _attribute_upsert(attribute_columns: list[dict]) -> str:
    """Return an UPSERT writing NEW's attributes, with ``{link_id}`` left to format."""
    if not attribute_columns:
        return ""
    names = ", ".join(_q(c["name"]) for c in attribute_columns)
    values = ", ".join(_new_value(c) for c in attribute_columns)
    assignments = ", ".join(f"{_q(c['name'])} = excluded.{_q(c['name'])}" for c in attribute_columns)
    return (
        f"INSERT INTO {LINK_ATTRIBUTES_TABLE} (link_id, {names}) VALUES ({{link_id}}, {values}) "
        f"ON CONFLICT (link_id) DO UPDATE SET {assignments};"
    )


def _trigger(suffix: str, event: str, body: list[str], when: str | None = None) -> str:
    when_clause = f" WHEN {when}" if when else ""
    statements = " ".join(body)
    head = f"CREATE TRIGGER {_TRIGGER_PREFIX}_{suffix} INSTEAD OF {event} ON {LINKS_VIEW}{when_clause}"
    return f"{head} BEGIN {statements} END"


def _register_spatial_view(conn: Connection) -> None:
    """Register links_full as a writable SpatiaLite view, so QGIS can load and edit it."""
    columns = {c["name"] for c in _table_info(conn, "views_geometry_columns")}
    if not columns:
        logger.warning(f"No views_geometry_columns table. {LINKS_VIEW} will not be registered as a spatial layer")
        return

    conn.execute("DELETE FROM views_geometry_columns WHERE view_name = ?", (LINKS_VIEW,))
    names = ["view_name", "view_geometry", "view_rowid", "f_table_name", "f_geometry_column"]
    values = [LINKS_VIEW, "geometry", "ogc_fid", LINKS_TABLE, "geometry"]
    if "read_only" in columns:  # Absent from metadata older than SpatiaLite 4
        names.append("read_only")
        values.append(0)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO views_geometry_columns ({', '.join(names)}) VALUES ({placeholders})", values)


def _view_is_current(conn: Connection) -> bool:
    try:
        view_columns = [c["name"].lower() for c in _table_info(conn, LINKS_VIEW)]
    except OperationalError:  # The view references a column that no longer exists
        return False
    if not view_columns:
        return False

    base = [c["name"].lower() for c in _table_info(conn, LINKS_TABLE)]
    attributes = [c["name"].lower() for c in _table_info(conn, LINK_ATTRIBUTES_TABLE)]
    attributes = [name for name in attributes if name not in base]
    return set(view_columns) == set(base) | set(attributes)


def _table_info(conn: Connection, table: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    return [{"name": r[1], "type": r[2], "notnull": r[3], "default": r[4]} for r in rows]


def _relation_exists(conn: Connection, name: str, kind: str) -> bool:
    sql = "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ? COLLATE NOCASE"
    return conn.execute(sql, (kind, name)).fetchone() is not None


def _new_value(column: dict) -> str:
    value = f"NEW.{_q(column['name'])}"
    return value if column["default"] is None else f"COALESCE({value}, {column['default']})"


def _slug(name: str) -> str:
    return re.sub(r"\W", "_", name).lower()


def _q(name: str) -> str:
    return escape_identifier(name)
