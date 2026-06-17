"""Strict invariant: the importer never issues ``ALTER TABLE`` on links/nodes.

We snoop ``sqlite_master.sql`` before and after the import and assert byte-equality.
"""

import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point


def _snapshot(path):
    with sqlite3.connect(path) as conn:
        return {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('links','nodes')"
            )
        }


def test_no_alter_table_links_or_nodes(empty_project):
    before = _snapshot(empty_project.path_to_file)

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

    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)

    after = _snapshot(empty_project.path_to_file)
    assert before == after, "Importer modified links/nodes schema (it must not)"
