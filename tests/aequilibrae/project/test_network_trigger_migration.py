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
        ORDER BY type, name
        """
    ).fetchall()


def _structure(conn):
    return [row for row in _schema(conn) if row[0] != "trigger" or not row[1].startswith("aequilibrae_")]


def _insert_link(conn, link_id, coordinates, a_node=0, b_node=0):
    conn.execute(
        """
        INSERT INTO links (link_id, a_node, b_node, modes, link_type, geometry)
        VALUES (?, ?, ?, 'c', 'default', GeomFromWKB(?, 4326))
        """,
        (link_id, a_node, b_node, LineString(coordinates).wkb),
    )


def test_migration_replaces_outdated_triggers(empty_project, migrate):
    """Test that migration 010 restores every current trigger and otherwise leaves a new project unchanged."""
    with empty_project.db_connection as conn:
        expected = _schema(conn)
        conn.execute("DROP TRIGGER aequilibrae_updated_link_geometry")
        conn.execute(
            "CREATE TRIGGER aequilibrae_updated_link_geometry AFTER UPDATE OF geometry ON links BEGIN SELECT 1; END"
        )
        conn.execute("DROP TRIGGER aequilibrae_turn_restrictions_keep_referenced_node")

        migrate(project_conn=conn)

        assert _schema(conn) == expected


def test_migration_preserves_project_structure(empty_project, migrate):
    """Test that migration 010 preserves existing tables, constraints, indexes and user triggers."""
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
        ]:
            conn.execute(f"CREATE INDEX {name} ON {table} ({column})")
        conn.execute("CREATE TRIGGER user_link_insert AFTER INSERT ON links BEGIN SELECT 1; END")
        expected = _structure(conn)

        migrate(project_conn=conn)

        assert _structure(conn) == expected


def test_migration_refreshes_stale_node_attributes(empty_project, migrate):
    """Test that migration 010 recomputes node modes and link types that earlier triggers left stale."""
    with empty_project.db_connection as conn:
        _insert_link(conn, 1001, [(0, 0), (1, 0)])
        expected = conn.execute("SELECT node_id, modes, link_types FROM nodes ORDER BY node_id").fetchall()
        remove_triggers(conn, "network")
        conn.execute("UPDATE nodes SET modes = 'cw', link_types = NULL")

        migrate(project_conn=conn)

        assert conn.execute("SELECT node_id, modes, link_types FROM nodes ORDER BY node_id").fetchall() == expected


@pytest.mark.parametrize("field", ["a_node", "b_node"])
@pytest.mark.parametrize("endpoint", [None, "invalid"])
def test_migration_preserves_endpoint_checks(empty_project, migrate, field, endpoint):
    """Test that migration 010 leaves the existing integer endpoint constraints in force."""
    with empty_project.db_connection as conn:
        for upgraded in (False, True):
            if upgraded:
                migrate(project_conn=conn)
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                _insert_link(conn, 1001, [(0, 0), (1, 0)], **{field: endpoint})

        _insert_link(conn, 1001, [(0, 0), (1, 0)])
        assert conn.execute("SELECT count(*) FROM links WHERE a_node IS NULL OR b_node IS NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_upgrade_preserves_project_structure(empty_project):
    """Test that upgrading a populated project preserves its tables, constraints and indexes."""
    with empty_project.db_connection as conn:
        _insert_link(conn, 1001, [(0, 0), (1, 0)])
        expected = _structure(conn)
        links = conn.execute("SELECT * FROM links ORDER BY link_id").fetchall()
        conn.execute("UPDATE migrations SET status = 'MISSING' WHERE id = 10")

    empty_project.upgrade()

    with empty_project.db_connection as conn:
        assert _structure(conn) == expected
        assert conn.execute("SELECT * FROM links ORDER BY link_id").fetchall() == links
        assert conn.execute("SELECT status FROM migrations WHERE id = 10").fetchone()[0] == "APPLIED"
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_upgraded_example_matches_new_project_triggers(sioux_falls_example, empty_project):
    """Test that an upgraded example project ends up with the same AequilibraE triggers as a new project."""

    def triggers(project):
        with project.db_connection as conn:
            sql = (
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'aequilibrae%' ORDER BY name"
            )
            return conn.execute(sql).fetchall()

    assert triggers(sioux_falls_example) == triggers(empty_project)
