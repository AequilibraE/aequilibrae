"""Tests for the schema-aware committer (``SpatialiteWriter._split_attributes``)."""

import json
import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point


def _basic_inputs():
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001, 10002, 10003],
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
            "a_node": [10000, 10001, 10002],
            "b_node": [10001, 10002, 10003],
            "direction": [0, 0, 0],
            "modes": ["c", "c", "c"],
            "link_type": ["residential", "residential", "primary"],
            "distance": [111000.0, 111000.0, 111000.0],
            "name": ["Street A", "Street B", "Avenue X"],
            "surface": ["asphalt", "gravel", "asphalt"],
            "bridge": ["yes", None, None],
            "_source_id": ["1", "2", "3"],   # IR scratch — should be stripped
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 1), (1, 1)]),
                LineString([(1, 1), (1, 0)]),
            ],
        },
        crs="EPSG:4326",
    )
    return nodes, links


def test_known_columns_land_in_real_columns(empty_project):
    nodes, links = _basic_inputs()
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM links")}
        assert names == {"Street A", "Street B", "Avenue X"}


def test_unknown_columns_land_in_other_attributes(empty_project):
    nodes, links = _basic_inputs()
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for link_id, oa in conn.execute(
            "SELECT link_id, other_attributes FROM links ORDER BY link_id"
        ):
            payload = json.loads(oa) if oa else {}
            assert "surface" in payload, f"link {link_id} missing 'surface'"
            # NaN/None should be dropped from the JSON object
            if link_id == 1:
                assert payload.get("bridge") == "yes"
            else:
                assert "bridge" not in payload


def test_nodes_unknown_columns_land_in_other_attributes(empty_project):
    nodes, links = _basic_inputs()
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for node_id, oa in conn.execute("SELECT node_id, other_attributes FROM nodes"):
            payload = json.loads(oa) if oa else {}
            assert "name" in payload, f"node {node_id} missing 'name'"
            assert "custom_node_attr" in payload, f"node {node_id} missing 'custom_node_attr'"


def test_underscore_prefixed_columns_are_stripped(empty_project):
    nodes, links = _basic_inputs()
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        # _source_id must not appear as a real column (we never ALTER) and must not
        # appear in other_attributes either.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(links)")]
        assert "_source_id" not in cols
        for (oa,) in conn.execute("SELECT other_attributes FROM links"):
            if oa:
                assert "_source_id" not in json.loads(oa)


def test_user_can_promote_attribute_by_adding_column(empty_project):
    """If the user adds a column manually before importing, it must be used."""
    with empty_project.db_connection as conn:
        conn.execute("ALTER TABLE links ADD COLUMN surface TEXT")

    nodes, links = _basic_inputs()
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        surfaces = {r[0] for r in conn.execute("SELECT surface FROM links")}
        assert surfaces == {"asphalt", "gravel"}
        # And surface must NOT appear in other_attributes any more
        for (oa,) in conn.execute("SELECT other_attributes FROM links"):
            payload = json.loads(oa) if oa else {}
            assert "surface" not in payload


def test_existing_other_attributes_is_merged_not_overwritten(empty_project):
    nodes, links = _basic_inputs()
    # Pre-supply a partial other_attributes column
    links = links.copy()
    links["other_attributes"] = [
        json.dumps({"pre_existing": "yes", "surface": "OVERRIDE_ME"}),
        None,
        None,
    ]
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(
            conn.execute("SELECT link_id, other_attributes FROM links ORDER BY link_id")
        )
        first = json.loads(rows[0][1])
        # Pre-existing key survived
        assert first["pre_existing"] == "yes"
        # The extras (surface, bridge) overrode the pre-existing duplicate
        assert first["surface"] == "asphalt"
