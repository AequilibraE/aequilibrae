"""Direction-alignment and length-computation helper tests."""

import math

import geopandas as gpd
from shapely.geometry import LineString

from aequilibrae.project.network.importer.utils import (
    aligned_along_geometry,
    compute_lengths,
)


def _arc(scale=1.0, reverse=False):
    ts = [i * (1.5 * math.pi) / 16 for i in range(17)]  # 0..270 degrees
    coords = [(math.cos(t) * scale, math.sin(t) * scale) for t in ts]
    if reverse:
        coords = coords[::-1]
    return LineString(coords)


def test_alignment_straight_lines():
    a = LineString([(0, 0), (0, 1), (0, 2)])
    same = LineString([(0.0001, 0), (0.0001, 1), (0.0001, 2)])
    rev = LineString([(0, 2), (0, 1), (0, 0)])
    assert aligned_along_geometry(a, same) is True
    assert aligned_along_geometry(a, rev) is False


def test_alignment_on_curved_arc_not_fooled_by_global_endpoints():
    arc = _arc()
    arc_same = _arc(scale=1.0001)
    arc_rev = _arc(scale=1.0001, reverse=True)
    assert aligned_along_geometry(arc, arc_same) is True
    assert aligned_along_geometry(arc, arc_rev) is False


def test_compute_lengths_small_extent_uses_utm():
    gs = gpd.GeoSeries([LineString([(0, 0), (0, 0.01)])], crs="EPSG:4326")
    length = compute_lengths(gs).iloc[0]
    # ~1.11 km for 0.01 deg of latitude
    assert 1100 < length < 1115


def test_compute_lengths_large_extent_uses_geodesic():
    # 10 degrees of longitude at 40N -> single UTM would distort badly.
    gs = gpd.GeoSeries([LineString([(-100, 40), (-90, 40)])], crs="EPSG:4326")
    length = compute_lengths(gs).iloc[0]
    assert 800_000 < length < 900_000
