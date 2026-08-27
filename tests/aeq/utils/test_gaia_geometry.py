"""Tests for the pure-Python SpatiaLite geometry BLOB codec.

The reference hex strings below were produced by mod_spatialite 5.1.0
(``GeomFromText`` / ``CompressGeometry``). They pin the codec to the exact
on-disk format so databases stay interchangeable with native SpatiaLite tools.
"""

import shapely
import shapely.wkt
import pytest

from aequilibrae.utils.gaia_geometry import (
    gaia_mbr,
    gaia_point_xy,
    gaia_srid,
    gaia_to_shapely,
    gaia_to_wkb,
    is_gaia_blob,
    linestring_boundary_point,
    make_point_blob,
    shapely_to_gaia,
    wkb_to_gaia,
)

# Uncompressed blobs written by mod_spatialite (GeomFromText(wkt, 4326))
REFERENCE_BLOBS = {
    "POINT(-96.77042 43.61283)": "0001e6100000cc0bb08f4e3158c0fc00a43671ce4540cc0bb08f4e3158c0fc00a43671ce45407c"
    "01000000cc0bb08f4e3158c0fc00a43671ce4540fe",
    "LINESTRING(-96.77042 43.61283, -96.711251 43.605813, -96.7 43.6)": "0001e6100000cc0bb08f4e3158c0cdcccccccccc"
    "4540cdcccccccc2c58c0fc00a43671ce45407c0200000003000000cc0bb08f4e3158c0fc00a43671ce4540d40fea22852d58c0ee3ec747"
    "8bcd4540cdcccccccc2c58c0cdcccccccccc4540fe",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.2 0.2, 0.4 0.2, 0.4 0.4, 0.2 0.4, 0.2 0.2)))": "0001e61000000000"
    "0000000000000000000000000000000000000000f03f000000000000f03f7c0600000001000000690300000002000000050000000000000"
    "0000000000000000000000000000000000000f03f0000000000000000000000000000f03f000000000000f03f00000000000000000000000"
    "00000f03f00000000000000000000000000000000050000009a9999999999c93f9a9999999999c93f9a9999999999d93f9a9999999999c9"
    "3f9a9999999999d93f9a9999999999d93f9a9999999999c93f9a9999999999d93f9a9999999999c93f9a9999999999c93ffe",
}

# (compressed blob, expected WKB) pairs produced by CompressGeometry/UncompressGeometry
COMPRESSED_BLOBS = {
    "linestring": (
        "0001e6100000cc0bb08f4e3158c0cdcccccccccc4540cdcccccccc2c58c0fc00a43671ce45407c42420f0003000000cc0bb08f4e"
        "3158c0fc00a43671ce4540315b723dddeee5bbcdcccccccc2c58c0cdcccccccccc4540fe",
        "010200000003000000cc0bb08f4e3158c0fc00a43671ce4540cc0bec22852d58c0fc00c7478bcd4540cdcccccccc2c58c0cdcccc"
        "cccccc4540",
    ),
    "polygon": (
        "0001e610000000000000000000000000000000000000000000000000f03f000000000000f03f7c43420f00010000000500000000"
        "0000000000000000000000000000000000803f00000000000000000000803f000080bf00000000000000000000000000000000000"
        "00000fe",
        "0103000000010000000500000000000000000000000000000000000000000000000000f03f000000000000000000000000000"
        "0f03f000000000000f03f0000000000000000000000000000f03f00000000000000000000000000000000",
    ),
}

ROUND_TRIP_WKTS = [
    "POINT(1.5 -2.5)",
    "LINESTRING(0 0, 1 1, 2 0.5)",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
    "MULTIPOINT(1 1, 2 2)",
    "MULTILINESTRING((0 0, 1 1), (2 2, 3 3, 4 4))",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))",
    "GEOMETRYCOLLECTION(POINT(1 1), LINESTRING(0 0, 2 2))",
    "POINT Z(1 2 3)",
    "LINESTRING Z(0 0 1, 1 1 2)",
]


@pytest.mark.parametrize("wkt,hex_blob", REFERENCE_BLOBS.items())
def test_byte_identical_with_native_spatialite(wkt, hex_blob):
    blob = bytes.fromhex(hex_blob)
    geom = shapely.wkt.loads(wkt)
    assert shapely_to_gaia(geom, 4326) == blob
    assert shapely.equals_exact(gaia_to_shapely(blob), geom)
    assert gaia_srid(blob) == 4326


@pytest.mark.parametrize("wkt", ROUND_TRIP_WKTS)
def test_round_trip(wkt):
    geom = shapely.wkt.loads(wkt)
    blob = shapely_to_gaia(geom, 4326)
    assert is_gaia_blob(blob)
    assert shapely.equals_exact(gaia_to_shapely(blob), geom)
    assert shapely.equals_exact(shapely.from_wkb(gaia_to_wkb(blob)), geom)


def test_wkb_to_gaia_round_trip():
    geom = shapely.LineString([(0, 0), (1, 1)])
    blob = wkb_to_gaia(geom.wkb, 4326)
    assert gaia_to_shapely(blob) == geom
    assert gaia_srid(blob) == 4326


@pytest.mark.parametrize("name,pair", COMPRESSED_BLOBS.items())
def test_reads_compressed_geometries(name, pair):
    blob, expected_wkb = bytes.fromhex(pair[0]), bytes.fromhex(pair[1])
    assert gaia_to_wkb(blob) == expected_wkb


def test_compressed_boundary_points():
    blob = bytes.fromhex(COMPRESSED_BLOBS["linestring"][0])
    assert gaia_point_xy(linestring_boundary_point(blob, True)) == (-96.77042, 43.61283)
    assert gaia_point_xy(linestring_boundary_point(blob, False)) == (-96.7, 43.6)


def test_fast_paths():
    blob = make_point_blob(-96.77042, 43.61283, 4326)
    assert blob == bytes.fromhex(REFERENCE_BLOBS["POINT(-96.77042 43.61283)"])
    assert gaia_point_xy(blob) == (-96.77042, 43.61283)
    assert gaia_mbr(blob) == (-96.77042, 43.61283, -96.77042, 43.61283)

    line = shapely.LineString([(0, 0), (1, 1), (2, 0.5)])
    lblob = shapely_to_gaia(line, 4326)
    assert gaia_mbr(lblob) == (0, 0, 2, 1)
    assert gaia_point_xy(linestring_boundary_point(lblob, True)) == (0, 0)
    assert gaia_point_xy(linestring_boundary_point(lblob, False)) == (2, 0.5)
    # StartPoint of a linestring must be byte-identical to an independently created point,
    # or the geometry-equality joins in the network triggers stop matching
    assert linestring_boundary_point(lblob, True) == make_point_blob(0, 0, 4326)


def test_boundary_point_of_non_linestring_is_none():
    blob = make_point_blob(1, 2, 4326)
    assert linestring_boundary_point(blob, True) is None


def test_is_gaia_blob():
    assert not is_gaia_blob(b"not a blob")
    assert not is_gaia_blob(None)
    assert not is_gaia_blob(shapely.Point(0, 0).wkb)
    assert is_gaia_blob(make_point_blob(0, 0, 4326))
