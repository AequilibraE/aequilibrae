from random import randint, random

import pytest
import shapely.wkb
from shapely.geometry import Point

from aequilibrae.utils.db_utils import read_and_close


def test_get(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    nd = randint(1, 24)
    node = nodes.get(nd)
    assert node.node_id == nd, "get node returned wrong object"
    nodes.renumber(nd, 200)
    with pytest.raises(ValueError, match=rf"nodes has no record with node_id={nd}"):
        _ = nodes.get(nd)


def test_save(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    chosen = [randint(1, 24) for _ in range(5)]
    while len(chosen) != len(set(chosen)):
        chosen = [randint(1, 24) for _ in range(5)]
    coords = []
    for nd in chosen:
        node = nodes.get(nd)
        x = node.geometry.x + random()
        y = node.geometry.y + random()
        coords.append([x, y])
        nodes.update(nd, is_centroid=0, geometry=Point([x, y]))
    for nd, crd in zip(chosen, coords, strict=True):
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


def test_lonlat(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    coordinates = nodes.lonlat.set_index("node_id")
    node = nodes.get(coordinates.index[0])
    assert coordinates.loc[node.node_id, "lon"] == pytest.approx(node.geometry.x)
    assert coordinates.loc[node.node_id, "lat"] == pytest.approx(node.geometry.y)


def test_connect_mode_rejects_regular_nodes(sioux_falls_example, caplog):
    nodes = sioux_falls_example.network.nodes
    node_id = next(iter(nodes)).node_id
    nodes.update(node_id, is_centroid=0)

    nodes.connect_mode(node_id, "c")

    assert "only makes sense for centroids" in caplog.text


def test_new_centroid(sioux_falls_example):
    nodes = sioux_falls_example.network.nodes
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'geometry'"):
        _ = nodes.new_centroid(1)
    tot_prev_centr = sioux_falls_example.network.count_centroids()
    tot_prev_nodes = sioux_falls_example.network.count_nodes()
    node_id = nodes.new_centroid(100, Point(1, 1))
    assert nodes.get(node_id).is_centroid == 1, "Creating new centroid returned wrong is_centroid value"
    assert sioux_falls_example.network.count_centroids() == tot_prev_centr + 1, "Failed to add centroids"
    assert sioux_falls_example.network.count_nodes() == tot_prev_nodes + 1, "Failed to add centroids"
