"""Building-footprint download gating tests for the neatnet exclusion mask."""

import geopandas as gpd
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.buildings import fetch_building_footprints
from aequilibrae.project.network.importer.staged_network import StagedNetwork


class _DummyCache:
    relative_path = None

    def write_geoparquet(self, name, gdf):
        return None


def _net(span_degrees):
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [100000, 100001],
            "geometry": [Point(0, 0), Point(span_degrees, span_degrees)],
            "modes": ["c", "c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [100000],
            "b_node": [100001],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["primary"],
            "distance": [111000.0],
            "geometry": [LineString([(0, 0), (span_degrees, span_degrees)])],
        },
        crs="EPSG:4326",
    )
    return StagedNetwork(nodes=nodes, links=links)


def test_buildings_disabled_by_default():
    assert fetch_building_footprints(_net(0.1), _DummyCache()) is None


def test_buildings_skipped_for_large_bbox_even_when_enabled():
    # Span well above the limit must skip the download (returns None) without
    # ever importing overturemaps.
    assert fetch_building_footprints(_net(5.0), _DummyCache(), enabled=True) is None
