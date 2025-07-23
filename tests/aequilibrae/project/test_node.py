from random import randint, random
from sqlite3 import IntegrityError

import shapely.wkb
from shapely.geometry import Point
import pytest

from aequilibrae.utils.db_utils import read_and_close


def test_save_and_assignment(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    nd = randint(1, 24)
    node = nodes.get(nd)

    with pytest.raises(AttributeError):
        node.modes = "abc"
    with pytest.raises(AttributeError):
        node.link_types = "default"
    with pytest.raises(AttributeError):
        node.node_id = 2
    with pytest.raises(ValueError):
        node.is_centroid = 2

    node.is_centroid = 0
    assert node.is_centroid == 0, "Assignment of is_centroid did not work"

    x = node.geometry.x + random()
    y = node.geometry.y + random()
    node.geometry = Point([x, y])
    node.save()

    with read_and_close(sioux_falls_example.path_to_file, spatial=True) as conn:
        sql = f"Select is_centroid, asBinary(geometry) from nodes where node_id={nd};"
        flag, wkb = conn.execute(sql).fetchone()
        assert flag == 0, "Saving of is_centroid failed"
        geo = shapely.wkb.loads(wkb)
        assert geo.x == x, "Geometry X saved wrong"
        assert geo.y == y, "Geometry Y saved wrong"

        sql = f"Select asBinary(geometry) from links where a_node={nd};"
        wkb = conn.execute(sql).fetchone()[0]
        geo2 = shapely.wkb.loads(wkb)
        assert geo2.xy[0][0] == x, "Saving node geometry broke underlying network"
        assert geo2.xy[1][0] == y, "Saving node geometry broke underlying network"


def test_data_fields(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    node1 = nodes.get(randint(1, 24))
    node2 = nodes.get(randint(1, 24))
    assert node1.data_fields() == node2.data_fields(), "Different nodes have different data fields"

    with read_and_close(sioux_falls_example.path_to_file) as conn:
        dt = conn.execute("pragma table_info(nodes)").fetchall()
        actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])
    fields = sorted(node1.data_fields())
    assert fields == actual_fields, "Node has unexpected set of fields"


def test_renumber(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    node = nodes.get(randint(2, 24))
    x = node.geometry.x
    y = node.geometry.y

    with pytest.raises(IntegrityError):
        node.renumber(1)

    num = randint(25, 2000)
    node.renumber(num)

    with read_and_close(sioux_falls_example.path_to_file, spatial=True) as conn:
        sql = f"Select asBinary(geometry) from nodes where node_id={num};"
        wkb = conn.execute(sql).fetchone()[0]
    geo = shapely.wkb.loads(wkb)
    assert geo.x == x, "Renumbering failed"
    assert geo.y == y, "Renumbering failed"
