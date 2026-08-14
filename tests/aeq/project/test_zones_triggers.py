import inspect
import sqlite3
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Point

from aequilibrae.project import about


@pytest.fixture
def queries():
    qry = Path(inspect.getfile(about)).parent / "database_specification/network/triggers/zones_triggers.sql"
    with open(qry, "r") as sql_file:
        queries = sql_file.read()
    return list(queries.split("#"))


def add_zone(conn, zone_id: int, x=0.0, y=0.0):
    geo = MultiPolygon([Point(x, y).buffer(0.01)])
    conn.execute("INSERT INTO zones (zone_id, geometry) VALUES(?, GeomFromWKB(?, 4326))", [zone_id, geo.wkb])


def add_node(conn, node_id: int, is_centroid=0, x=0.0, y=0.0):
    sql = "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES(?, ?, GeomFromWKB(?, 4326))"
    conn.execute(sql, [node_id, is_centroid, Point(x, y).wkb])


def is_centroid(conn, node_id: int):
    return conn.execute("SELECT is_centroid FROM nodes WHERE node_id=?", [node_id]).fetchone()


def test_all_tests_considered(queries):
    """Test that every trigger in the zones triggers specification file has a test of its own."""
    import sys

    current_module = sys.modules[__name__]
    tests_added = dir(current_module)
    tests_added = [x[5:] for x in tests_added if x[:5] == "test_"]

    for trigger in queries:
        if "TRIGGER" in trigger.upper():
            found = [x for x in tests_added if x in trigger]
            if not found:
                pytest.fail(f"Trigger not tested. {trigger}")


def test_new_node_zone_centroid(empty_project):
    """Test that a node created with the ID of an existing zone is tagged as a centroid."""
    with empty_project.db_connection as conn:
        add_zone(conn, 1)
        add_node(conn, 1)
        assert is_centroid(conn, 1) == (1,), "Node created over an existing zone was not tagged as a centroid"

        add_node(conn, 2, x=1.0)
        assert is_centroid(conn, 2) == (0,), "Node without a matching zone was tagged as a centroid"


def test_updated_node_id_zone_centroid(empty_project):
    """Test that a node renumbered to the ID of an existing zone is tagged as a centroid."""
    with empty_project.db_connection as conn:
        add_zone(conn, 10)
        add_node(conn, 55)
        assert is_centroid(conn, 55) == (0,), "Node without a matching zone was tagged as a centroid"

        conn.execute("UPDATE nodes SET node_id=10 WHERE node_id=55")
        assert is_centroid(conn, 10) == (1,), "Node renumbered onto an existing zone was not tagged as a centroid"


def test_nodes_iscentroid_zone_update(empty_project):
    """Test that a centroid cannot be untagged while the zone it belongs to still exists."""
    with empty_project.db_connection as conn:
        add_zone(conn, 1)
        add_node(conn, 1, is_centroid=1)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE nodes SET is_centroid=0 WHERE node_id=1")
        assert is_centroid(conn, 1) == (1,), "Centroid was untagged while its zone still existed"

        # Centroids that do not match a zone are not protected, so the existing triggers delete this one
        add_node(conn, 2, is_centroid=1, x=1.0)
        conn.execute("UPDATE nodes SET is_centroid=0 WHERE node_id=2")
        assert is_centroid(conn, 2) is None, "Centroid without a matching zone could not be untagged"


def test_new_zone_centroid(empty_project):
    """Test that creating a zone over an existing node tags that node as the zone's centroid."""
    with empty_project.db_connection as conn:
        add_node(conn, 7)
        assert is_centroid(conn, 7) == (0,), "Node without a matching zone was tagged as a centroid"

        add_zone(conn, 7)
        assert is_centroid(conn, 7) == (1,), "Node was not tagged as a centroid when its zone was created"


def test_updated_zone_id_centroid(empty_project):
    """Test that renumbering a zone onto an existing node tags that node as the zone's centroid."""
    with empty_project.db_connection as conn:
        add_node(conn, 5)
        add_zone(conn, 50)
        assert is_centroid(conn, 5) == (0,), "Node without a matching zone was tagged as a centroid"

        conn.execute("UPDATE zones SET zone_id=5 WHERE zone_id=50")
        assert is_centroid(conn, 5) == (1,), "Node was not tagged as a centroid when its zone was renumbered onto it"


def test_zone_centroid_triggers_added_with_zoning_layer(empty_project):
    """Test that recreating the zoning layer brings back the triggers that tag centroids."""
    tables = [
        "zones",
        "idx_zones_geometry",
        "idx_zones_geometry_node",
        "idx_zones_geometry_parent",
        "idx_zones_geometry_rowid",
    ]
    with empty_project.db_connection as conn:
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.execute("DELETE FROM attributes_documentation WHERE name_table LIKE 'zones'")

    empty_project.zoning.create_zoning_layer()

    with empty_project.db_connection as conn:
        add_zone(conn, 3)
        add_node(conn, 3)
        assert is_centroid(conn, 3) == (1,), "Zone centroid triggers were not recreated with the zoning layer"


def test_zone_centroid_migration(sioux_falls_example):
    """Test that upgrading a project tags every node that shares its ID with a zone as a centroid."""
    with sioux_falls_example.db_connection as conn:
        # Nodes 1 and 2 have links attached to them, so untagging them does not delete them
        conn.execute("UPDATE nodes SET is_centroid=0 WHERE node_id in (1, 2)")
        assert is_centroid(conn, 1) == (0,), "Could not untag the centroid before upgrading the project"

    sioux_falls_example.upgrade(ignore_transit=True, ignore_results=True)

    with sioux_falls_example.db_connection as conn:
        untagged = conn.execute("SELECT count(*) FROM nodes WHERE is_centroid != 1").fetchone()[0]
        assert untagged == 0, "Upgrading the project did not tag the nodes that share their ID with a zone"

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE nodes SET is_centroid=0 WHERE node_id=1")
