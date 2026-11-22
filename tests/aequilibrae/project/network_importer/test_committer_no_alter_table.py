import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.db_writer import SpatialiteWriter
from aequilibrae.project.network.importer.staged_network import StagedNetwork


def _snapshot(path):
    with sqlite3.connect(path) as conn:
        tables = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('links','nodes')"
            )
        }
        attributes = [r[1] for r in conn.execute("PRAGMA table_info(link_attributes)")]
    return tables, attributes


def test_no_alter_table_links_or_nodes(empty_project):
    """Test that importing never alters links or nodes, and gives link attributes no field but other_attributes."""
    before, attributes_before = _snapshot(empty_project.path_to_file)

    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001],
            "geometry": [Point(0, 0), Point(0, 1)],
            "modes": ["c", "c"],
            "made_up_attr": ["x", "y"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [10000],
            "b_node": [10001],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["residential"],
            "distance": [111000.0],
            "name": ["Street A"],
            "lots_of": ["values"],
            "should_not": ["alter"],
            "the_schema": ["ever"],
            "geometry": [LineString([(0, 0), (0, 1)])],
        },
        crs="EPSG:4326",
    )

    SpatialiteWriter(empty_project).write(StagedNetwork(nodes=nodes, links=links))

    after, attributes_after = _snapshot(empty_project.path_to_file)
    assert before == after
    # Attributes without a field all go to other_attributes, which new projects do not have
    assert attributes_after == [*attributes_before, "other_attributes"]
