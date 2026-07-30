"""Robustness tests for the Spatialite writer: trigger restoration and link-type folding."""

import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.db_writer import SpatialiteWriter
from aequilibrae.project.network.importer.staged_network import StagedNetwork


def _count_triggers(path):
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0])


def _staged(n_link_types=1):
    n = n_link_types + 1
    node_ids = list(range(100000, 100000 + n))
    nodes = gpd.GeoDataFrame(
        {
            "node_id": node_ids,
            "geometry": [Point(0, i) for i in range(n)],
            "modes": ["c"] * n,
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": list(range(1, n_link_types + 1)),
            "a_node": node_ids[:-1],
            "b_node": node_ids[1:],
            "direction": [0] * n_link_types,
            "modes": ["c"] * n_link_types,
            "link_type": [f"type_{i}" for i in range(n_link_types)],
            "distance": [111000.0] * n_link_types,
            "geometry": [LineString([(0, i), (0, i + 1)]) for i in range(n_link_types)],
        },
        crs="EPSG:4326",
    )
    net = StagedNetwork(nodes=nodes, links=links)
    net.validate()
    return net


def test_triggers_restored_after_successful_write(empty_project):
    before = _count_triggers(empty_project.path_to_file)
    SpatialiteWriter(empty_project).write(_staged(1))
    after = _count_triggers(empty_project.path_to_file)
    assert after >= before


def test_excess_link_types_fold_into_other_link_types(empty_project):
    # The schema only allows ~50 single-char link_type_ids; importing far more
    # must fold the least-frequent into a catch-all rather than raising.
    net = _staged(n_link_types=100)
    # Make type_0 the most frequent so it is retained, not folded.
    links = net.links.copy()
    links.loc[links["link_type"] != "type_0", "link_type"] = links.loc[
        links["link_type"] != "type_0", "link_type"
    ]
    SpatialiteWriter(empty_project).write(net)

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_type_ids = [r[0] for r in conn.execute("SELECT link_type_id FROM link_types")]
        assert all(len(code) == 1 for code in link_type_ids)
        used = {r[0] for r in conn.execute("SELECT DISTINCT link_type FROM links")}
        assert "other_link_types" in used
