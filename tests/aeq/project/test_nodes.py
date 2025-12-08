from copy import copy, deepcopy
from random import randint, random

import shapely.wkb
from shapely.geometry import Point
import pytest

from aequilibrae.utils.db_utils import read_and_close


def test_get(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    nd = randint(1, 24)
    node = nodes.get(nd)
    assert node.node_id == nd, "get node returned wrong object"
    node.renumber(200)
    with pytest.raises(ValueError):
        _ = nodes.get(nd)


def test_save(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    chosen = [randint(1, 24) for _ in range(5)]
    while len(chosen) != len(set(chosen)):
        chosen = [randint(1, 24) for _ in range(5)]
    coords = []
    for nd in chosen:
        node = nodes.get(nd)
        node.is_centroid = 0
        x = node.geometry.x + random()
        y = node.geometry.y + random()
        coords.append([x, y])
        node.geometry = Point([x, y])
    nodes.save()
    for nd, crd in zip(chosen, coords):
        x, y = crd
        with read_and_close(sioux_falls_example.path_to_file, spatial=True) as conn:
            sql = f"Select is_centroid, asBinary(geometry) from nodes where node_id={nd};"
            flag, wkb = conn.execute(sql).fetchone()
        assert flag == 0, "Saving of is_centroid failed"
        geo = shapely.wkb.loads(wkb)
        assert geo.x == x, "Geometry X saved wrong"
        assert geo.y == y, "Geometry Y saved wrong"


def test_fields(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    f_editor = nodes.fields
    fields = sorted(f_editor.all_fields())
    with read_and_close(sioux_falls_example.path_to_file) as conn:
        dt = conn.execute("pragma table_info(nodes)").fetchall()
    actual_fields = sorted({x[1] for x in dt if x[1] != "ogc_fid"})
    assert fields == actual_fields, "Table editor is weird for table nodes"


def test_copy(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    with pytest.raises(Exception):
        _ = copy(nodes)
    with pytest.raises(Exception):
        _ = deepcopy(nodes)


def test_new_centroid(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    with pytest.raises(Exception):
        _ = nodes.new_centroid(1)
    tot_prev_centr = sioux_falls_example.network.count_centroids()
    tot_prev_nodes = sioux_falls_example.network.count_nodes()
    node = nodes.new_centroid(100)
    assert node.is_centroid == 1, "Creating new centroid returned wrong is_centroid value"
    node.geometry = Point(1, 1)
    node.save()
    assert sioux_falls_example.network.count_centroids() == tot_prev_centr + 1, "Failed to add centroids"
    assert sioux_falls_example.network.count_nodes() == tot_prev_nodes + 1, "Failed to add centroids"
