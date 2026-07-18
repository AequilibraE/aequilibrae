"""Building-footprint download gating tests for the neatnet exclusion mask."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.buildings import BuildingMaskResult, fetch_building_footprints
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


def test_buildings_disabled_explicitly():
    result = fetch_building_footprints(_net(0.1), _DummyCache(), enabled=False)
    assert isinstance(result, BuildingMaskResult)
    assert result.gdf is None
    assert result.status == "disabled"
    assert result.attempted is False


def test_buildings_skipped_for_large_bbox_even_when_enabled():
    # Span well above the limit must skip the download without ever importing overturemaps.
    result = fetch_building_footprints(_net(5.0), _DummyCache(), enabled=True)
    assert result.gdf is None
    assert result.status == "skipped"
    assert result.reason == "bbox_guard"


def test_buildings_retry_once_then_fallback(monkeypatch):
    net = _net(0.1)

    class _FailingOverture:
        def __init__(self):
            self.calls = 0

        def record_batch_reader(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("boom")

    fake = _FailingOverture()
    monkeypatch.setattr("aequilibrae.project.network.importer.buildings.require", lambda *a, **k: fake)
    monkeypatch.setattr(
        "aequilibrae.project.network.importer.sources.overture.impl.get_latest_overture_version",
        lambda: "test-release",
    )

    result = fetch_building_footprints(net, _DummyCache(), enabled=True)
    assert result.gdf is None
    assert result.status == "fallback"
    assert result.retries == 1
    assert result.reason == "RuntimeError"
    assert fake.calls == 2


def _patch_release(monkeypatch):
    monkeypatch.setattr(
        "aequilibrae.project.network.importer.sources.overture.impl.get_latest_overture_version",
        lambda: "test-release",
    )


def test_buildings_success_path_downloads_and_caches(monkeypatch):
    pa = pytest.importorskip("pyarrow")
    from shapely import to_wkb
    from shapely.geometry import Polygon

    footprint = Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])
    table = pa.table({"id": ["b1"], "geometry": [to_wkb(footprint)]})

    class _Reader:
        def read_all(self):
            return table

    class _FakeOverture:
        def record_batch_reader(self, *args, **kwargs):
            return _Reader()

    class _RecordingCache:
        def __init__(self):
            self.written = []

        def write_geoparquet(self, name, gdf):
            self.written.append((name, len(gdf)))

    monkeypatch.setattr("aequilibrae.project.network.importer.buildings.require", lambda *a, **k: _FakeOverture())
    _patch_release(monkeypatch)

    cache = _RecordingCache()
    result = fetch_building_footprints(_net(0.1), cache, enabled=True)
    assert result.status == "downloaded"
    assert result.cache_written is True
    assert result.retries == 0
    assert len(result.gdf) == 1
    assert str(result.gdf.crs).upper() == "EPSG:4326"
    assert cache.written == [("buildings.parquet", 1)]


def test_buildings_zero_rows_twice_falls_back(monkeypatch):
    pa = pytest.importorskip("pyarrow")

    class _Reader:
        def read_all(self):
            return pa.table({"id": pa.array([], type=pa.string())})

    class _FakeOverture:
        def record_batch_reader(self, *args, **kwargs):
            return _Reader()

    monkeypatch.setattr("aequilibrae.project.network.importer.buildings.require", lambda *a, **k: _FakeOverture())
    _patch_release(monkeypatch)

    result = fetch_building_footprints(_net(0.1), _DummyCache(), enabled=True)
    assert result.gdf is None
    assert result.status == "fallback"
    assert result.reason == "zero_rows"
    assert result.retries == 1
