import sqlite3

import pytest
from shapely.geometry import LineString, Point


def _insert_link(conn, link_id, coordinates, modes="c", link_type="default"):
    conn.execute(
        """
        INSERT INTO links (link_id, a_node, b_node, modes, link_type, geometry)
        VALUES (?, 0, 0, ?, ?, GeomFromWKB(?, 4326))
        """,
        (link_id, modes, link_type, LineString(coordinates).wkb),
    )
    return conn.execute("SELECT a_node, b_node FROM links WHERE link_id = ?", (link_id,)).fetchone()


def _build_adjacent_network(project):
    with project.db_connection as conn:
        moving_node, replaced_node = _insert_link(conn, 1001, [(0, 0), (1, 0)])
        shared_node, other_node = _insert_link(
            conn,
            1002,
            [(1, 0), (2, 0)],
            modes="w",
            link_type="centroid_connector",
        )
        assert shared_node == replaced_node
    return moving_node, replaced_node, other_node


def _network_snapshot(conn):
    nodes = conn.execute("SELECT node_id, is_centroid, Hex(geometry) FROM nodes ORDER BY node_id").fetchall()
    links = conn.execute(
        "SELECT link_id, a_node, b_node, Hex(geometry), distance FROM links ORDER BY link_id"
    ).fetchall()
    return nodes, links


def _assert_network_consistent(conn):
    dangling = conn.execute(
        """
        SELECT count(*)
        FROM links AS link
        WHERE NOT EXISTS (SELECT 1 FROM nodes WHERE node_id = link.a_node)
           OR NOT EXISTS (SELECT 1 FROM nodes WHERE node_id = link.b_node)
        """
    ).fetchone()[0]
    mismatched = conn.execute(
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
    indexes = conn.execute(
        "SELECT CheckSpatialIndex('nodes', 'geometry'), CheckSpatialIndex('links', 'geometry')"
    ).fetchone()

    assert dangling == 0
    assert mismatched == 0
    assert indexes == (1, 1)


def test_delete_links_delete_nodes(sioux_falls_example):
    items = sioux_falls_example.network.count_nodes()
    assert items == 24, "Wrong number of nodes found"
    links = sioux_falls_example.network.links
    nodes = sioux_falls_example.network.nodes

    node = nodes.get(1)
    node.is_centroid = 0
    node.save()

    for i in [1, 2, 3, 4, 5, 14]:
        link = links.get(i)
        link.delete()
    items = sioux_falls_example.network.count_nodes()
    assert items == 23, "Wrong number of nodes found"


def test_add_regular_link(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        data = [123456, "c", "default", LineString([Point(0, 0), Point(1, 1)]).wkb]
        sql = "insert into links (link_id, modes, link_type, geometry) Values(?,?,?,GeomFromWKB(?, 4326));"
        conn.execute(sql, data)


def test_add_regular_node_change_centroid_id(sioux_falls_example):
    network = sioux_falls_example.network
    nodes_count = network.count_nodes()
    data = [987654, 1, Point(0, 0).wkb]

    with sioux_falls_example.db_connection as conn:
        sql = "insert into nodes (node_id, is_centroid, geometry) Values(?,?,GeomFromWKB(?, 4326));"
        conn.execute(sql, data)
        conn.commit()
        assert network.count_nodes() == nodes_count + 1, "Failed to insert node"

        conn.execute("Update nodes set is_centroid=0 where node_id=?", data[:1])
        conn.commit()
        assert network.count_nodes() == nodes_count, "Failed to delete node when changing centroid flag"


def test_link_direction(sioux_falls_example):
    network = sioux_falls_example.network
    links_count = network.count_links()

    with sioux_falls_example.db_connection as conn:
        sql = "UPDATE links SET direction=-2 WHERE link_id=1;"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)

        data = [987654, 2, "c", "default", LineString([Point(0, 0), Point(1, 0)]).wkb]
        sql_insert = (
            "insert into links (link_id, direction, modes, link_type, geometry) Values(?,?,?,?,GeomFromWKB(?, 4326));"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql_insert, data)

        data = [
            (987654, -1, "c", "default", LineString([Point(0, 0), Point(1, 0)]).wkb),
            (876543, 0, "c", "default", LineString([Point(1, 0), Point(1, 1)]).wkb),
            (765432, 1, "c", "default", LineString([Point(1, 1), Point(0, 1)]).wkb),
        ]
        conn.executemany(sql_insert, data)
        conn.commit()
        assert network.count_links() == links_count + 3, "Failed when adding new links to the project."


@pytest.mark.parametrize("field", ["a_node", "b_node"])
@pytest.mark.parametrize("bad_endpoint", ["opposite", "missing", "null"])
def test_link_endpoint_updates_are_guarded(empty_project, field, bad_endpoint):
    """Test that setting a_node/b_node to a node that is not the matching link endpoint is rejected."""
    with empty_project.db_connection as conn:
        a_node, b_node = _insert_link(conn, 1001, [(0, 0), (1, 0)])
        if bad_endpoint == "opposite":
            bad_node = b_node if field == "a_node" else a_node
        elif bad_endpoint == "missing":
            bad_node = conn.execute("SELECT max(node_id) + 100 FROM nodes").fetchone()[0]
        else:
            bad_node = None
        before = _network_snapshot(conn)

        endpoint = "start" if field == "a_node" else "end"
        with pytest.raises(sqlite3.IntegrityError, match=f"{field} does not match the {endpoint} point"):
            conn.execute(f"UPDATE links SET {field} = ? WHERE link_id = 1001", (bad_node,))

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)


def test_link_endpoint_noop_update_is_allowed(empty_project):
    """Test that rewriting a_node/b_node with their current values passes the endpoint guards."""
    with empty_project.db_connection as conn:
        _insert_link(conn, 1001, [(0, 0), (1, 0)])
        conn.execute("UPDATE links SET a_node = a_node, b_node = b_node WHERE link_id = 1001")
        _assert_network_consistent(conn)


def test_geometry_update_cannot_bypass_endpoint_guards(empty_project):
    """Test that changing geometry and a_node in a single statement cannot smuggle in a mismatched endpoint."""
    with empty_project.db_connection as conn:
        _, wrong_a_node = _insert_link(conn, 1001, [(0, 0), (1, 0)])
        before = _network_snapshot(conn)

        with pytest.raises(sqlite3.IntegrityError, match="a_node does not match the start point"):
            conn.execute(
                """
                UPDATE links
                SET a_node = ?, geometry = GeomFromWKB(?, 4326)
                WHERE link_id = 1001
                """,
                (wrong_a_node, LineString([(2, 0), (1, 0)]).wkb),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)


def test_link_geometry_update_rebuilds_endpoint_nodes(empty_project):
    """Test that moving a link's end point creates the new node and drops the orphaned old one."""
    with empty_project.db_connection as conn:
        a_node, old_b_node = _insert_link(conn, 1001, [(0, 0), (1, 0)])
        conn.execute(
            "UPDATE links SET geometry = GeomFromWKB(?, 4326) WHERE link_id = 1001",
            (LineString([(0, 0), (2, 0)]).wkb,),
        )
        new_a_node, new_b_node = conn.execute("SELECT a_node, b_node FROM links WHERE link_id = 1001").fetchone()

        assert new_a_node == a_node
        assert new_b_node != old_b_node
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (old_b_node,)).fetchone()[0] == 0
        _assert_network_consistent(conn)


def test_adjacent_node_merge_is_ordered_and_consistent(empty_project):
    """Test that dragging a node onto its neighbour merges them into a self-loop with merged modes/link types."""
    moving_node, replaced_node, other_node = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute(
            """
            UPDATE nodes
            SET geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
            WHERE node_id = ?
            """,
            (replaced_node, moving_node),
        )

        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (replaced_node,)).fetchone()[0] == 0
        self_loop = conn.execute("SELECT a_node, b_node, distance FROM links WHERE link_id = 1001").fetchone()
        continuing_link = conn.execute("SELECT a_node, b_node FROM links WHERE link_id = 1002").fetchone()
        assert self_loop[:2] == (moving_node, moving_node)
        assert self_loop[2] == pytest.approx(0)
        assert continuing_link == (moving_node, other_node)
        modes, link_types = conn.execute(
            "SELECT modes, link_types FROM nodes WHERE node_id = ?", (moving_node,)
        ).fetchone()
        expected_link_types = {
            row[0]
            for row in conn.execute(
                "SELECT link_type_id FROM link_types WHERE link_type IN ('default', 'centroid_connector')"
            )
        }
        assert set(modes) == {"c", "w"}
        assert set(link_types) == expected_link_types
        _assert_network_consistent(conn)


def test_adjacent_node_merge_is_consistent_in_reverse_direction(empty_project):
    """Test that the same neighbour merge stays consistent when the node dragged is the downstream one."""
    first_node, replaced_node, moving_node = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute(
            """
            UPDATE nodes
            SET geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
            WHERE node_id = ?
            """,
            (replaced_node, moving_node),
        )

        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (replaced_node,)).fetchone()[0] == 0
        assert conn.execute("SELECT a_node, b_node FROM links WHERE link_id = 1001").fetchone() == (
            first_node,
            moving_node,
        )
        assert conn.execute("SELECT a_node, b_node FROM links WHERE link_id = 1002").fetchone() == (
            moving_node,
            moving_node,
        )
        _assert_network_consistent(conn)


@pytest.mark.parametrize("moving_is_centroid,replaced_is_centroid", [(0, 1), (1, 0), (1, 1)])
def test_centroid_merge_is_atomic(empty_project, moving_is_centroid, replaced_is_centroid):
    """Test that a merge involving a centroid on either side is aborted and leaves the network untouched."""
    moving_node, replaced_node, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute("UPDATE nodes SET is_centroid = ? WHERE node_id = ?", (moving_is_centroid, moving_node))
        conn.execute("UPDATE nodes SET is_centroid = ? WHERE node_id = ?", (replaced_is_centroid, replaced_node))
        before = _network_snapshot(conn)

        with pytest.raises(sqlite3.IntegrityError, match="Cannot cannibalize centroids"):
            conn.execute(
                """
                UPDATE nodes
                SET geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
                WHERE node_id = ?
                """,
                (replaced_node, moving_node),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)


def test_centroid_demotion_cannot_bypass_merge_guard(empty_project):
    """Test that clearing is_centroid in the same statement as the move does not unlock the centroid merge guard."""
    moving_node, replaced_node, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute("UPDATE nodes SET is_centroid = 1 WHERE node_id = ?", (moving_node,))
        before = _network_snapshot(conn)

        with pytest.raises(sqlite3.IntegrityError, match="Cannot cannibalize centroids"):
            conn.execute(
                """
                UPDATE nodes
                SET is_centroid = 0,
                    geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
                WHERE node_id = ?
                """,
                (replaced_node, moving_node),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)


def test_linked_centroid_can_move_and_be_demoted(empty_project):
    """Test that a centroid attached to links can be demoted and moved to empty space in one statement."""
    moving_node, _, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute("UPDATE nodes SET is_centroid = 1 WHERE node_id = ?", (moving_node,))
        conn.execute(
            """
            UPDATE nodes
            SET is_centroid = 0, geometry = GeomFromWKB(?, 4326)
            WHERE node_id = ?
            """,
            (Point(-1, 0).wkb, moving_node),
        )

        assert conn.execute("SELECT is_centroid FROM nodes WHERE node_id = ?", (moving_node,)).fetchone() == (0,)
        _assert_network_consistent(conn)


def test_empty_centroid_must_be_demoted_separately_from_geometry(empty_project):
    """Test that a link-less centroid rejects a combined demotion plus move, but accepts each change on its own."""
    with empty_project.db_connection as conn:
        conn.execute(
            "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (9001, 1, GeomFromWKB(?, 4326))",
            (Point(9, 9).wkb,),
        )
        before = _network_snapshot(conn)

        with pytest.raises(sqlite3.IntegrityError, match="empty centroid must be demoted separately"):
            conn.execute(
                """
                UPDATE nodes
                SET is_centroid = 0, geometry = GeomFromWKB(?, 4326)
                WHERE node_id = 9001
                """,
                (Point(10, 10).wkb,),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)

        conn.execute(
            "UPDATE nodes SET geometry = GeomFromWKB(?, 4326) WHERE node_id = 9001",
            (Point(10, 10).wkb,),
        )
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = 9001").fetchone()[0] == 1
        _assert_network_consistent(conn)

        conn.execute("UPDATE nodes SET is_centroid = 0 WHERE node_id = 9001")
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = 9001").fetchone()[0] == 0
        _assert_network_consistent(conn)


def test_isolated_centroid_demotion_and_merge_is_atomic(empty_project):
    """Test that demoting a link-less centroid while dragging it onto another node aborts with no side effects."""
    _, replaced_node, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute(
            "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (9001, 1, GeomFromWKB(?, 4326))",
            (Point(9, 9).wkb,),
        )
        before = _network_snapshot(conn)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                UPDATE nodes
                SET is_centroid = 0,
                    geometry = (SELECT geometry FROM nodes WHERE node_id = ?)
                WHERE node_id = 9001
                """,
                (replaced_node,),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)


def test_node_id_and_geometry_actual_changes_must_be_separate(empty_project):
    """Test that renumbering a node and moving it must happen in separate statements, while no-op halves are allowed."""
    moving_node, _, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        before = _network_snapshot(conn)
        with pytest.raises(sqlite3.IntegrityError, match="node_id must be updated separately"):
            conn.execute(
                """
                UPDATE nodes
                SET node_id = 9001, geometry = GeomFromWKB(?, 4326)
                WHERE node_id = ?
                """,
                (Point(-1, 0).wkb, moving_node),
            )

        assert _network_snapshot(conn) == before
        _assert_network_consistent(conn)

        conn.execute(
            "UPDATE nodes SET node_id = 9001, geometry = geometry WHERE node_id = ?",
            (moving_node,),
        )
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (moving_node,)).fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = 9001").fetchone()[0] == 1

        conn.execute(
            """
            UPDATE nodes
            SET node_id = node_id, geometry = GeomFromWKB(?, 4326)
            WHERE node_id = 9001
            """,
            (Point(-1, 0).wkb,),
        )
        _assert_network_consistent(conn)


def test_linked_centroid_can_be_renumbered_and_demoted(empty_project):
    """Test that a centroid attached to links can be renumbered and demoted in one statement."""
    moving_node, _, _ = _build_adjacent_network(empty_project)

    with empty_project.db_connection as conn:
        conn.execute("UPDATE nodes SET is_centroid = 1 WHERE node_id = ?", (moving_node,))
        conn.execute(
            "UPDATE nodes SET node_id = 9001, is_centroid = 0 WHERE node_id = ?",
            (moving_node,),
        )

        assert conn.execute("SELECT count(*) FROM nodes WHERE node_id = ?", (moving_node,)).fetchone()[0] == 0
        assert conn.execute("SELECT is_centroid FROM nodes WHERE node_id = 9001").fetchone() == (0,)
        _assert_network_consistent(conn)
