import json
import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.db_writer import SpatialiteWriter
from aequilibrae.project.network.importer.staged_network import StagedNetwork


def _basic_inputs():
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [100000, 100001, 100002, 100003],
            "geometry": [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)],
            "modes": ["c", "c", "c", "c"],
            "custom_node_attr": ["n0", "n1", "n2", "n3"],
            "name": ["n0", "n1", "n2", "n3"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1, 2, 3],
            "a_node": [100000, 100001, 100002],
            "b_node": [100001, 100002, 100003],
            "direction": [0, 0, 0],
            "modes": ["c", "c", "c"],
            "link_type": ["residential", "residential", "primary"],
            "distance": [111000.0, 111000.0, 111000.0],
            "name": ["Street A", "Street B", "Avenue X"],
            "surface": ["asphalt", "gravel", "asphalt"],
            "bridge": ["yes", None, None],
            "source_id": ["1", "2", "3"],
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 1), (1, 1)]),
                LineString([(1, 1), (1, 0)]),
            ],
        },
        crs="EPSG:4326",
    )
    return nodes, links


def _write(empty_project, nodes, links):
    net = StagedNetwork(nodes=nodes, links=links)
    net.validate()
    SpatialiteWriter(empty_project).write(net)


def test_known_columns_land_in_real_columns(empty_project):
    nodes, links = _basic_inputs()
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM links")}
        assert names == {"Street A", "Street B", "Avenue X"}


def test_unknown_columns_land_in_other_attributes(empty_project):
    nodes, links = _basic_inputs()
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for link_id, other_attributes in conn.execute("SELECT link_id, other_attributes FROM links ORDER BY link_id"):
            payload = json.loads(other_attributes) if other_attributes else {}
            assert "surface" in payload
            assert "source_id" in payload
            if link_id == 1:
                assert payload.get("bridge") == "yes"
            else:
                assert "bridge" not in payload


def test_nodes_unknown_columns_land_in_other_attributes(empty_project):
    nodes, links = _basic_inputs()
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for _node_id, other_attributes in conn.execute("SELECT node_id, other_attributes FROM nodes"):
            payload = json.loads(other_attributes) if other_attributes else {}
            assert "name" in payload
            assert "custom_node_attr" in payload


def test_underscore_prefixed_columns_are_stripped(empty_project):
    nodes, links = _basic_inputs()
    links = links.assign(_scratch=["1", "2", "3"])
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(links)")]
        assert "_scratch" not in cols
        for (other_attributes,) in conn.execute("SELECT other_attributes FROM links"):
            if other_attributes:
                assert "_scratch" not in json.loads(other_attributes)


def test_user_can_promote_attribute_by_adding_column(empty_project):
    with empty_project.db_connection as conn:
        conn.execute("ALTER TABLE links ADD COLUMN surface TEXT")

    nodes, links = _basic_inputs()
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        surfaces = {r[0] for r in conn.execute("SELECT surface FROM links")}
        assert surfaces == {"asphalt", "gravel"}
        for (other_attributes,) in conn.execute("SELECT other_attributes FROM links"):
            payload = json.loads(other_attributes) if other_attributes else {}
            assert "surface" not in payload


def test_existing_other_attributes_is_merged_not_overwritten(empty_project):
    nodes, links = _basic_inputs()
    links = links.copy()
    links["other_attributes"] = [json.dumps({"pre_existing": "yes", "surface": "OVERRIDE_ME"}), None, None]
    _write(empty_project, nodes, links)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(conn.execute("SELECT link_id, other_attributes FROM links ORDER BY link_id"))
        first = json.loads(rows[0][1])
        assert first["pre_existing"] == "yes"
        assert first["surface"] == "asphalt"
