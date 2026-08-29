"""Unit tests for Overture schema translation (no overturemaps/pyarrow required)."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.sources.overture.schema_to_staged import (
    _direction_for_segment,
    _modes_for_segment,
    build_staged_from_overture,
)

_META = {
    "source": "overture",
    "backend": "cloud",
    "source_url": "s3://test",
    "release": "test-release",
    "fetched_at": "2026-07-18T00:00:00+00:00",
}


def _connectors(ids_points):
    return gpd.GeoDataFrame(
        {"id": [i for i, _ in ids_points], "geometry": [p for _, p in ids_points]},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _segments(rows):
    base = {"subtype": "road", "class": "residential"}
    records, geoms = [], []
    for row in rows:
        rec = dict(base)
        rec.update(row)
        geoms.append(rec.pop("geometry"))
        records.append(rec)
    return gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")


def test_missing_connector_is_synthesized_from_segment_geometry():
    connectors = _connectors([("c0", Point(0, 0))])
    segments = _segments(
        [
            {
                "id": "seg-0",
                "geometry": LineString([(0, 0), (0, 0.001)]),
                "connectors": [{"connector_id": "c0", "at": 0.0}, {"connector_id": "missing", "at": 1.0}],
            }
        ]
    )
    net = build_staged_from_overture(connectors=connectors, segments=segments, modes=("car",), source_meta=_META)
    net.validate()
    assert len(net.links) == 1
    assert set(net.nodes["source_id"]) == {"c0", "missing"}
    synth = net.nodes.loc[net.nodes["source_id"] == "missing", "geometry"].iloc[0]
    assert (round(synth.x, 6), round(synth.y, 6)) == (0.0, 0.001)


@pytest.mark.parametrize(
    "subtype, cls, expected",
    [
        ("road", "residential", "bctw"),
        ("road", "primary", "bctw"),
        ("road", "motorway", "c"),
        ("road", "trunk", "ct"),
        ("road", "service", "ctw"),
        ("road", "footway", "w"),
        ("road", "path", "bw"),
        ("road", "cycleway", "b"),
        ("road", "steps", "w"),
        ("rail", "rail", ""),
        ("water", "ferry", ""),
    ],
)
def test_modes_for_segment_by_class(subtype, cls, expected):
    assert _modes_for_segment({"subtype": subtype, "class": cls}) == expected


def _deny(heading):
    return {"access_type": "denied", "when": {"heading": heading}, "heading": None}


def test_direction_for_segment_variants():
    assert _direction_for_segment({"access_restrictions": None}) == 0
    assert _direction_for_segment({"access_restrictions": [_deny("backward")]}) == 1
    assert _direction_for_segment({"access_restrictions": [_deny("forward")]}) == -1
    assert _direction_for_segment({"access_restrictions": [_deny("forward"), _deny("backward")]}) == 0


def _seg(seg_id, connectors, geometry, **over):
    rec = {"id": seg_id, "subtype": "road", "class": "residential", "connectors": connectors}
    rec.update(over)
    rec["geometry"] = geometry
    return rec


def _degenerate_segments():
    """One usable segment plus the three malformed shapes seen in real Overture data."""
    rows = [
        _seg("good", [{"connector_id": "c0", "at": 0.0}, {"connector_id": "c1", "at": 1.0}],
             LineString([(0, 0), (0, 0.001)])),
        # Identical 'at' offsets: no consecutive pair can be split (the Quito case).
        _seg("no-splits", [{"connector_id": "c1", "at": 0.5}, {"connector_id": "c2", "at": 0.5}],
             LineString([(0, 0.001), (0, 0.002)])),
        _seg("one-connector", [{"connector_id": "c2", "at": 0.0}],
             LineString([(0, 0.002), (0, 0.003)])),
        _seg("empty-geom", [{"connector_id": "c0", "at": 0.0}, {"connector_id": "c1", "at": 1.0}],
             LineString()),
    ]
    geoms = [r.pop("geometry") for r in rows]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def _three_connectors():
    return _connectors([("c0", Point(0, 0)), ("c1", Point(0, 0.001)), ("c2", Point(0, 0.002))])


def test_malformed_segments_are_skipped_not_fatal():
    """One bad record must not abort a whole metro-sized import."""
    net = build_staged_from_overture(
        connectors=_three_connectors(), segments=_degenerate_segments(), modes=("car",), source_meta=_META
    )
    net.validate()
    assert len(net.links) == 1
    assert net.links.iloc[0]["source_id"] == "good"


def test_skipped_segments_are_reported(caplog):
    with caplog.at_level("WARNING"):
        build_staged_from_overture(
            connectors=_three_connectors(), segments=_degenerate_segments(), modes=("car",), source_meta=_META
        )
    message = "\n".join(caplog.messages)
    assert "Skipped 3 malformed Overture segments" in message
    for reason in ("empty_geometry", "no_valid_splits", "too_few_connectors"):
        assert reason in message


def test_all_segments_malformed_still_raises():
    """Skipping bad records must not silently produce an empty network."""
    with pytest.raises(ImporterError, match="no Overture links remain"):
        build_staged_from_overture(
            connectors=_three_connectors(),
            segments=_degenerate_segments().iloc[1:],
            modes=("car",),
            source_meta=_META,
        )
