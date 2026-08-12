import sqlite3

import pytest


ENDPOINT_GUARDS = {
    "aequilibrae_links_a_node_update",
    "aequilibrae_links_b_node_update",
}

LEGACY_TABLE_CONSTRAINT_TRIGGERS = {
    "link_type_single_letter_insert",
    "link_type_single_letter_update",
    "mode_single_letter_insert",
    "mode_single_letter_update",
    "modes_length_on_links_insert",
    "modes_length_on_links_update",
}

RETIRED_OR_REPLACED_LEGACY_TRIGGERS = {
    "aequilibrae_cannibalize_node",
    "aequilibrae_cannibalise_node",
    "cannibalize_node",
    "cannibalise_node",
    "cannibalize_node_abort_when_centroid",
    "default_period_delete",
    "default_period_update",
    "new_link",
    "new_link_a_node",
    "new_link_b_node",
    "update_link_a_node",
    "update_link_b_node",
    "updated_link_geometry",
    "deleted_link",
    "enforces_link_length_update",
    "links_direction_insert",
    "links_direction_update",
    "mode_keep_if_in_use_deleting",
    "mode_keep_if_in_use_updating",
    "modes_on_links_insert",
    "modes_on_links_update",
    "update_node_geometry",
    "no_duplicate_node",
    "dont_delete_node",
    "updated_node_id",
    "nodes_iscentroid_change_update",
    "nodes_iscentroid_insert",
    "nodes_iscentroid_update",
}


def _trigger_names(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}


def _assert_topology_and_spatial_indexes_are_consistent(conn):
    inconsistent_links = conn.execute(
        """
        SELECT count(*)
        FROM links AS link
        WHERE NOT EXISTS (
            SELECT 1
            FROM nodes AS node
            WHERE node.node_id = link.a_node
              AND node.geometry = StartPoint(link.geometry))
           OR NOT EXISTS (
            SELECT 1
            FROM nodes AS node
            WHERE node.node_id = link.b_node
              AND node.geometry = EndPoint(link.geometry))
        """
    ).fetchone()[0]

    assert inconsistent_links == 0
    assert conn.execute(
        "SELECT CheckSpatialIndex('nodes', 'geometry'), CheckSpatialIndex('links', 'geometry')"
    ).fetchone() == (1, 1)


def test_upgrade_repairs_and_protects_legacy_link_endpoints(sioux_falls_test):
    """Test that migration 003 repairs broken endpoints and indexes, swaps legacy triggers for the new guards."""
    with sioux_falls_test.db_connection as conn:
        conn.execute(
            """
            CREATE TRIGGER link_type_single_letter_insert
            BEFORE INSERT ON link_types
            WHEN length(new.link_type_id) != 1
            BEGIN
                SELECT RAISE(ABORT, 'Link type codes need to be a single letter');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER link_type_single_letter_update
            BEFORE UPDATE OF link_type_id ON link_types
            WHEN length(new.link_type_id) != 1
            BEGIN
                SELECT RAISE(ABORT, 'Link type codes need to be a single letter');
            END
            """
        )

        # Select an endpoint with exactly one geometrically matching node, which
        # makes the migration's repair deterministic.
        link_id, original_a_node = conn.execute(
            """
            SELECT link.link_id, link.a_node
            FROM links AS link
            WHERE 1 = (
                SELECT count(*)
                FROM nodes AS node
                WHERE node.geometry = StartPoint(link.geometry))
            ORDER BY link.link_id
            LIMIT 1
            """
        ).fetchone()
        expected_a_node = conn.execute(
            """
            SELECT node.node_id
            FROM nodes AS node, links AS link
            WHERE link.link_id = ?
              AND node.geometry = StartPoint(link.geometry)
            """,
            (link_id,),
        ).fetchone()[0]
        bad_a_node = conn.execute(
            """
            SELECT node_id
            FROM nodes
            WHERE node_id != ?
            ORDER BY node_id
            LIMIT 1
            """,
            (expected_a_node,),
        ).fetchone()[0]

        assert original_a_node == expected_a_node
        conn.execute("UPDATE links SET a_node = ? WHERE link_id = ?", (bad_a_node, link_id))
        assert conn.execute("SELECT a_node FROM links WHERE link_id = ?", (link_id,)).fetchone()[0] == bad_a_node
        expected_node_rowid = conn.execute("SELECT ROWID FROM nodes WHERE node_id = ?", (expected_a_node,)).fetchone()[
            0
        ]
        conn.execute("DELETE FROM idx_nodes_geometry WHERE pkid = ?", (expected_node_rowid,))
        assert conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone() == (0,)

    with pytest.warns(UserWarning, match="Take care when ignoring a database during an upgrade"):
        sioux_falls_test.upgrade(ignore_transit=True, ignore_results=True)

    with sioux_falls_test.db_connection as conn:
        assert conn.execute("SELECT status FROM migrations WHERE id = 3").fetchone() == ("APPLIED",)

        trigger_names = _trigger_names(conn)
        assert ENDPOINT_GUARDS <= trigger_names
        assert LEGACY_TABLE_CONSTRAINT_TRIGGERS <= trigger_names
        assert RETIRED_OR_REPLACED_LEGACY_TRIGGERS.isdisjoint(trigger_names)
        assert conn.execute("SELECT a_node FROM links WHERE link_id = ?", (link_id,)).fetchone()[0] == expected_a_node

        _assert_topology_and_spatial_indexes_are_consistent(conn)

    with pytest.raises(
        sqlite3.IntegrityError,
        match="a_node does not match the start point of link geometry",
    ):
        with sioux_falls_test.db_connection as conn:
            conn.execute("UPDATE links SET a_node = ? WHERE link_id = ?", (bad_a_node, link_id))

    with sioux_falls_test.db_connection as conn:
        assert conn.execute("SELECT a_node FROM links WHERE link_id = ?", (link_id,)).fetchone()[0] == expected_a_node

        moving_node, replaced_node = conn.execute(
            "SELECT a_node, b_node FROM links WHERE a_node != b_node ORDER BY link_id LIMIT 1"
        ).fetchone()
        conn.execute(
            "UPDATE nodes SET is_centroid = 0 WHERE node_id IN (?, ?)",
            (moving_node, replaced_node),
        )
        conn.execute(
            """
            UPDATE nodes
            SET geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
            WHERE node_id = ?
            """,
            (replaced_node, moving_node),
        )

        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (moving_node,)).fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (replaced_node,)).fetchone()[0] == 0
        _assert_topology_and_spatial_indexes_are_consistent(conn)


def test_new_project_has_protected_schema_without_running_migration(empty_project):
    """Test that a freshly created project already ships the endpoint guards and marks migration 003 as SKIPPED."""
    with empty_project.db_connection as conn:
        assert conn.execute("SELECT status FROM migrations WHERE id = 3").fetchone() == ("SKIPPED",)
        assert ENDPOINT_GUARDS <= _trigger_names(conn)
        _assert_topology_and_spatial_indexes_are_consistent(conn)


def test_irreparable_legacy_endpoint_rolls_back_migration(sioux_falls_test):
    """Test that an endpoint with no matching node aborts migration 003 and rolls the schema back untouched."""
    with sioux_falls_test.db_connection as conn:
        conn.execute("DROP TRIGGER dont_delete_node")
        missing_node = conn.execute("SELECT a_node FROM links ORDER BY link_id LIMIT 1").fetchone()[0]
        conn.execute("DELETE FROM nodes WHERE node_id = ?", (missing_node,))
        triggers_before_upgrade = _trigger_names(conn)

    with pytest.warns(UserWarning, match="Take care when ignoring a database during an upgrade"):
        with pytest.raises(RuntimeError, match="some endpoints have no unique matching node"):
            sioux_falls_test.upgrade(ignore_transit=True, ignore_results=True)

    with sioux_falls_test.db_connection as conn:
        assert conn.execute("SELECT status FROM migrations WHERE id = 3").fetchone() == ("MISSING",)
        assert _trigger_names(conn) == triggers_before_upgrade
        assert ENDPOINT_GUARDS.isdisjoint(_trigger_names(conn))
