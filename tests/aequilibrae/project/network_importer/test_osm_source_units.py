"""Unit tests for OSM source frame-preparation helpers (no pyrosm/osmnx required)."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.sources.osm.impl import _first_last_points, _pyrosm_nodes_frame


def test_first_last_points_linestring():
    first, last = _first_last_points(LineString([(0, 0), (1, 1), (2, 0)]))
    assert (first.x, first.y) == (0, 0)
    assert (last.x, last.y) == (2, 0)


def test_first_last_points_multilinestring_spans_parts():
    geom = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 5)]])
    first, last = _first_last_points(geom)
    assert (first.x, first.y) == (0, 0)
    assert (last.x, last.y) == (2, 5)


def test_first_last_points_unsupported_geometry():
    assert _first_last_points(None) == (None, None)
    assert _first_last_points(Point(1, 1)) == (None, None)


def _nodes_frame(ids, points):
    return gpd.GeoDataFrame({"id": ids, "geometry": points}, geometry="geometry", crs="EPSG:4326")


def test_pyrosm_nodes_frame_keeps_only_used_nodes():
    nodes = _nodes_frame([10, 11, 12], [Point(0, 0), Point(1, 0), Point(9, 9)])
    edges = gpd.GeoDataFrame(
        {"u": [10], "v": [11], "geometry": [LineString([(0, 0), (1, 0)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    out = _pyrosm_nodes_frame(nodes, edges)
    assert set(out["osm_id"]) == {10, 11}


def test_pyrosm_nodes_frame_synthesizes_missing_nodes_from_edge_endpoints():
    # Node 99 is referenced by an edge but absent from the nodes frame; its
    # position must be recovered from the edge geometry's end point.
    nodes = _nodes_frame([10], [Point(0, 0)])
    edges = gpd.GeoDataFrame(
        {"u": [10], "v": [99], "geometry": [LineString([(0, 0), (2, 3)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    out = _pyrosm_nodes_frame(nodes, edges)
    assert set(out["osm_id"]) == {10, 99}
    synth = out.loc[out["osm_id"] == 99, "geometry"].iloc[0]
    assert (synth.x, synth.y) == (2, 3)


def test_pyrosm_nodes_frame_requires_uv_columns():
    nodes = _nodes_frame([10], [Point(0, 0)])
    edges = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 0)])]}, geometry="geometry", crs="EPSG:4326")
    with pytest.raises(ImporterError, match="OSM node ids"):
        _pyrosm_nodes_frame(nodes, edges)
