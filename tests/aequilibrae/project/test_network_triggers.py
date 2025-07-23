import sqlite3
import pytest
from shapely.geometry import LineString, Point


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
