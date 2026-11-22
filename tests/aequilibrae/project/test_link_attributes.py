import sqlite3

import pytest
from shapely.geometry import LineString

from aequilibrae.project.network.links_view import MANDATORY_LINK_FIELDS, links_source

# links as created before link attributes moved to their own table: no constraints at all
LEGACY_LINKS = """CREATE TABLE links (ogc_fid INTEGER PRIMARY KEY, link_id INTEGER UNIQUE, a_node INTEGER,
    b_node INTEGER, direction INTEGER NOT NULL DEFAULT 0, distance NUMERIC, modes TEXT NOT NULL,
    link_type TEXT NOT NULL DEFAULT 'link type not defined', name TEXT, speed_ab NUMERIC, toll REAL DEFAULT 1.5)"""

# links as created by the specification that preceded the split, with its CHECKs attached to other_attributes
CURRENT_SPEC_LINKS = """CREATE TABLE links (ogc_fid INTEGER PRIMARY KEY, link_id INTEGER NOT NULL UNIQUE,
    a_node INTEGER, b_node INTEGER, direction INTEGER NOT NULL DEFAULT 0, distance NUMERIC, modes TEXT NOT NULL,
    link_type TEXT REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT, 'name' TEXT,
    speed_ab NUMERIC, toll REAL DEFAULT 1.5, other_attributes TEXT
    CHECK(TYPEOF(link_id) == 'integer') CHECK(direction IN (-1, 0, 1)))"""


def _columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _add_link(conn, link_id, x=0.0):
    sql = "INSERT INTO links (link_id, modes, link_type, geometry) VALUES (?, 'c', 'default', GeomFromText(?, 4326))"
    conn.execute(sql, (link_id, f"LINESTRING({x} 0, {x + 0.01} 0)"))


def _make_pre_split(project, links_ddl):
    """Turn an empty project back into one whose links table holds every link field, with two links."""
    with project.db_connection as conn:
        conn.execute("DROP VIEW links_full")
        conn.execute("DELETE FROM views_geometry_columns WHERE view_name = 'links_full'")
        sql = "SELECT name FROM sqlite_master WHERE type = 'trigger' AND sql LIKE '%link_attributes%'"
        for (trigger,) in conn.execute(sql).fetchall():
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("DROP TABLE link_attributes")
        conn.execute("SELECT DisableSpatialIndex('links', 'geometry')")
        conn.execute("DROP TABLE idx_links_geometry")
        conn.execute("SELECT DiscardGeometryColumn('links', 'geometry')")
        conn.execute("DROP TABLE links")

        conn.execute(links_ddl)
        conn.execute("SELECT AddGeometryColumn('links', 'geometry', 4326, 'LINESTRING', 'XY')")
        conn.execute("SELECT CreateSpatialIndex('links', 'geometry')")
        conn.execute("CREATE INDEX idx_links_speed ON links (speed_ab)")
        rows = [
            (7, 101, 1, 2, "First", 50.0, "LINESTRING(0 0, 0.01 0)"),
            (9, 102, 2, 3, "Second", 60.0, "LINESTRING(0.01 0, 0.02 0)"),
        ]
        conn.executemany(
            "INSERT INTO links (ogc_fid, link_id, a_node, b_node, modes, link_type, name, speed_ab, geometry) "
            "VALUES (?, ?, ?, ?, 'c', 'default', ?, ?, GeomFromText(?, 4326))",
            rows,
        )
        conn.execute("DELETE FROM migrations WHERE id = 12")


def test_new_project_splits_links(empty_project):
    """Test that a new project keeps only the mandatory link fields in links, behind a registered links_full view."""
    with empty_project.db_connection as conn:
        assert _columns(conn, "links") == list(MANDATORY_LINK_FIELDS)
        assert "speed_ab" in _columns(conn, "link_attributes")
        registration = conn.execute("SELECT view_rowid, f_table_name, read_only FROM views_geometry_columns").fetchall()
    assert registration == [("ogc_fid", "links", 0)]
    assert empty_project.tables.links._source == "links_full"


def test_link_attributes_follow_links(empty_project):
    """Test that adding, renumbering and deleting a link creates, renumbers and deletes its attribute record."""
    with empty_project.db_connection as conn:
        _add_link(conn, 10)
        assert conn.execute("SELECT link_id FROM link_attributes").fetchall() == [(10,)]

        conn.execute("UPDATE links SET link_id = 11 WHERE link_id = 10")
        assert conn.execute("SELECT link_id FROM link_attributes").fetchall() == [(11,)]

        conn.execute("DELETE FROM links WHERE link_id = 11")
        assert conn.execute("SELECT count(*) FROM link_attributes").fetchone()[0] == 0


def test_link_attributes_cannot_be_changed_on_their_own(empty_project):
    """Test that attribute records cannot be deleted, orphaned or renumbered apart from their link."""
    with empty_project.db_connection as conn:
        _add_link(conn, 10)
        with pytest.raises(sqlite3.IntegrityError, match="deleted with their links"):
            conn.execute("DELETE FROM link_attributes WHERE link_id = 10")
        with pytest.raises(sqlite3.IntegrityError, match="require an existing link_id"):
            conn.execute("INSERT INTO link_attributes (link_id) VALUES (99)")
        with pytest.raises(sqlite3.IntegrityError, match="follows links.link_id"):
            conn.execute("UPDATE link_attributes SET link_id = 99 WHERE link_id = 10")


def test_links_full_insert_and_delete(empty_project):
    """Test that inserting into links_full writes both tables with column defaults, and deleting removes both."""
    with empty_project.db_connection as conn:
        conn.execute(
            "INSERT INTO links_full (link_id, modes, link_type, speed_ab, name, geometry) "
            "VALUES (20, 'c', 'default', 80, 'Main', GeomFromText('LINESTRING(0 0, 0.01 0)', 4326))"
        )
        assert conn.execute("SELECT direction FROM links WHERE link_id = 20").fetchone() == (0,)
        assert conn.execute("SELECT speed_ab, name FROM link_attributes WHERE link_id = 20").fetchone() == (80, "Main")

        conn.execute("DELETE FROM links_full WHERE link_id = 20")
        assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM link_attributes").fetchone()[0] == 0


def test_links_full_updates_only_touch_their_own_columns(empty_project):
    """Test that attribute edits through links_full never update links, and links edits only set their column."""
    with empty_project.db_connection as conn:
        _add_link(conn, 10)
        conn.execute("CREATE TABLE update_log (column_name TEXT)")
        for column in ("modes", "geometry"):
            conn.execute(
                f"CREATE TRIGGER log_{column} AFTER UPDATE OF {column} ON links "
                f"BEGIN INSERT INTO update_log VALUES ('{column}'); END"
            )

        conn.execute("UPDATE links_full SET speed_ab = 50, capacity_ab = 900 WHERE link_id = 10")
        assert conn.execute("SELECT * FROM update_log").fetchall() == []
        assert conn.execute("SELECT speed_ab, capacity_ab FROM link_attributes").fetchone() == (50, 900)

        conn.execute("UPDATE links_full SET modes = 'ct' WHERE link_id = 10")
        assert conn.execute("SELECT * FROM update_log").fetchall() == [("modes",)]

        with pytest.raises(sqlite3.IntegrityError, match="ogc_fid cannot be changed"):
            conn.execute("UPDATE links_full SET ogc_fid = 999 WHERE link_id = 10")


def test_links_api_writes_through_links_full(empty_project):
    """Test that the links API reads and writes both tables, and raises for links that do not exist."""
    links = empty_project.tables.links
    links.insert(link_id=10, modes="c", speed_ab=50.0, geometry=LineString([(0, 0), (0.01, 0)]))
    links.update(10, capacity_ab=900.0, modes="ct")

    record = links.get(10)
    assert (record.speed_ab, record.capacity_ab, record.modes) == (50.0, 900.0, "ct")
    with pytest.raises(ValueError, match="no record with link_id=99"):
        links.update(99, speed_ab=1.0)
    with pytest.raises(ValueError, match="no record with link_id=99"):
        links.delete(99)


def test_new_link_fields_go_to_link_attributes(empty_project):
    """Test that fields added through the links API are created in link_attributes and shown by links_full."""
    links = empty_project.tables.links
    links.insert(link_id=10, modes="c", geometry=LineString([(0, 0), (0.01, 0)]))
    links.fields.add("lanes", "number of lanes", "INTEGER", directional=True)
    links.update(10, lanes_ab=3)

    with empty_project.db_connection as conn:
        assert "lanes_ab" not in _columns(conn, "links")
        assert {"lanes_ab", "lanes_ba"} <= set(_columns(conn, "link_attributes"))
        assert conn.execute("SELECT lanes_ab FROM links_full WHERE link_id = 10").fetchone() == (3,)
    assert links.fields.get_description("lanes_*") == "number of lanes"


def test_stale_links_full_is_rebuilt(empty_project):
    """Test that links_full is rebuilt when link_attributes gained a column outside AequilibraE."""
    with empty_project.db_connection as conn:
        conn.execute("ALTER TABLE link_attributes ADD COLUMN surface TEXT")
        assert "surface" not in _columns(conn, "links_full")
        assert links_source(conn) == "links_full"
        assert "surface" in _columns(conn, "links_full")


@pytest.mark.parametrize("links_ddl", [LEGACY_LINKS, CURRENT_SPEC_LINKS], ids=["legacy", "current_spec"])
def test_migration_splits_links(empty_project, links_ddl):
    """Test that upgrading moves every non-mandatory field and its index to link_attributes, keeping ogc_fids."""
    _make_pre_split(empty_project, links_ddl)
    empty_project.tables.refresh_schemas()
    assert empty_project.tables.links._source == "links"

    empty_project.upgrade()

    with empty_project.db_connection as conn:
        assert _columns(conn, "links") == list(MANDATORY_LINK_FIELDS)
        rows = conn.execute("SELECT link_id, name, speed_ab, toll FROM link_attributes ORDER BY link_id").fetchall()
        assert rows == [(101, "First", 50.0, 1.5), (102, "Second", 60.0, 1.5)]
        assert conn.execute("SELECT ogc_fid FROM links ORDER BY link_id").fetchall() == [(7,), (9,)]
        index = conn.execute("SELECT tbl_name FROM sqlite_master WHERE name = 'idx_links_speed'").fetchone()
        assert index == ("link_attributes",)
        links_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'links'").fetchone()[0]

    # Each table keeps its own constraints: none for the legacy one, the specification's for the other
    assert ("CHECK" in links_sql.upper()) == (links_ddl == CURRENT_SPEC_LINKS)
    assert empty_project.tables.links.get(101).speed_ab == 50.0
