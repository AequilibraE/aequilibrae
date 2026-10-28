import sqlite3
from pathlib import Path

import pytest
from shapely.geometry import LineString

from aequilibrae.project.project_creation import remove_triggers
from aequilibrae.utils.model_run_utils import import_file_as_module

MIGRATION = (
    Path(__file__).parents[3]
    / "src"
    / "aequilibrae"
    / "project"
    / "database_specification"
    / "network"
    / "migrations"
    / "010_refresh_network_triggers.py"
)


@pytest.fixture
def migrate():
    return import_file_as_module(MIGRATION, "test_migration_010_refresh_network_triggers", force=True).migrate


def _schema(conn):
    return conn.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('index', 'trigger')
          AND tbl_name IN ('links', 'nodes', 'modes', 'link_types', 'turn_restrictions', 'periods', 'scenarios')
        ORDER BY type, name
        """
    ).fetchall()


def _indexed_columns(conn, table):
    return sorted(
        tuple(row[2] for row in conn.execute(f"PRAGMA index_info('{index[1]}')"))
        for index in conn.execute(f"PRAGMA index_list({table})")
    )


def _insert_link_without_nodes(conn, link_id, coordinates):
    conn.execute(
        "INSERT INTO links (link_id, modes, link_type, geometry) VALUES (?, 'c', 'default', GeomFromWKB(?, 4326))",
        (link_id, LineString(coordinates).wkb),
    )


def _restore_strict_endpoint_checks(conn):
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'links'").fetchone()[0]
    for field in ("a_node", "b_node"):
        sql = sql.replace(f"CHECK(TYPEOF({field}) IN ('integer', 'null'))", f"CHECK(TYPEOF({field}) == 'integer')")
    version = conn.execute("PRAGMA schema_version").fetchone()[0]
    conn.execute("PRAGMA writable_schema = ON")
    conn.execute("UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = 'links'", (sql,))
    conn.execute(f"PRAGMA schema_version = {version + 1}")
    conn.execute("PRAGMA writable_schema = OFF")


def test_migration_replaces_outdated_triggers(empty_project, migrate):
    """Test that migration 010 restores every current trigger and otherwise leaves a new project unchanged."""
    with empty_project.db_connection as conn:
        expected = _schema(conn)
        conn.execute("DROP TRIGGER aequilibrae_updated_link_geometry")
        conn.execute("CREATE TRIGGER aequilibrae_updated_link_geometry AFTER UPDATE OF geometry ON links BEGIN SELECT 1; END")
        conn.execute("DROP TRIGGER aequilibrae_turn_restrictions_keep_referenced_node")

        migrate(project_conn=conn)

        assert _schema(conn) == expected


def test_migration_drops_redundant_indexes(empty_project, migrate):
    """Test that migration 010 keeps a single index per column where earlier specifications created duplicates."""
    with empty_project.db_connection as conn:
        conn.execute("DROP INDEX idx_link_anode")
        for name, table, column in [
            ("aequilibrae_links_a_node_idx", "links", "a_node"),
            ("links_a_node_idx", "links", "a_node"),
            ("aequilibrae_links_b_node_idx", "links", "b_node"),
            ("links_b_node_idx", "links", "b_node"),
            ("aequilibrae_links_link_type", "links", "link_type"),
            ("links_link_type", "links", "link_type"),
            ("aequilibrae_nodes_node_id", "nodes", "node_id"),
            ("nodes_node_id", "nodes", "node_id"),
            ("idx_node", "nodes", "node_id"),
        ]:
            conn.execute(f"CREATE INDEX {name} ON {table} ({column})")
        conn.execute("CREATE UNIQUE INDEX idx_link ON links (link_id)")

        migrate(project_conn=conn)

        assert _indexed_columns(conn, "links") == [
            ("a_node",),
            ("a_node", "b_node"),
            ("b_node",),
            ("link_id",),
            ("link_type",),
            ("modes",),
        ]
        assert _indexed_columns(conn, "nodes") == [("is_centroid",), ("node_id",)]
        link_indexes = {row[1]: row[2] for row in conn.execute("PRAGMA index_list(links)")}
        assert "idx_link_anode" in link_indexes
        assert [unique for name, unique in link_indexes.items() if name.startswith("sqlite_autoindex")] == [1]


def test_migration_refreshes_stale_node_attributes(empty_project, migrate):
    """Test that migration 010 recomputes node modes and link types that earlier triggers left stale."""
    with empty_project.db_connection as conn:
        _insert_link_without_nodes(conn, 1001, [(0, 0), (1, 0)])
        expected = conn.execute("SELECT node_id, modes, link_types FROM nodes ORDER BY node_id").fetchall()
        remove_triggers(conn, "network")
        conn.execute("UPDATE nodes SET modes = 'cw', link_types = NULL")

        migrate(project_conn=conn)

        assert conn.execute("SELECT node_id, modes, link_types FROM nodes ORDER BY node_id").fetchall() == expected


def test_migration_allows_links_without_endpoint_nodes(empty_project, migrate):
    """Test that migration 010 relaxes the a_node/b_node type checks so links can be added without them."""
    with empty_project.db_connection as conn:
        _restore_strict_endpoint_checks(conn)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            _insert_link_without_nodes(conn, 1001, [(0, 0), (1, 0)])

        migrate(project_conn=conn)

        _insert_link_without_nodes(conn, 1001, [(0, 0), (1, 0)])
        assert conn.execute("SELECT count(*) FROM links WHERE a_node IS NULL OR b_node IS NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_upgraded_example_matches_new_project_triggers(sioux_falls_example, empty_project):
    """Test that an upgraded example project ends up with the same AequilibraE triggers as a new project."""

    def triggers(project):
        with project.db_connection as conn:
            sql = "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'aequilibrae%' ORDER BY name"
            return conn.execute(sql).fetchall()

    assert triggers(sioux_falls_example) == triggers(empty_project)
