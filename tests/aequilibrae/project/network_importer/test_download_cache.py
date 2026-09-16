"""Direct tests for the ``DownloadCache`` helper."""

import hashlib
import json

import geopandas as gpd
from shapely.geometry import Point

from aequilibrae.project.network.importer import DownloadCache


def test_cache_is_lazy(tmp_path):
    """No folder is created unless something is written."""
    cache = DownloadCache(tmp_path, "osm-overpass", "vatican")
    assert cache.relative_path is None
    assert not (tmp_path / "downloaded data").exists()


def test_write_geoparquet_creates_folder_and_round_trips(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "test")
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    path = cache.write_geoparquet("payload", gdf)
    assert path.exists()
    assert path.suffix == ".parquet"
    assert cache.relative_path is not None
    assert "downloaded data" in cache.relative_path
    assert "osm-overpass" in cache.relative_path
    assert "test" in cache.relative_path

    back = gpd.read_parquet(path)
    assert len(back) == 2
    assert list(back["name"]) == ["a", "b"]




def test_write_json_round_trips(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "test")
    path = cache.write_json("notes", {"hello": "world"})
    assert path.exists()
    assert path.suffix == ".json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"hello": "world"}


def test_manifest_includes_sha256(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "test")
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    written = cache.write_geoparquet("payload", gdf)
    cache.write_manifest({"bbox": [0, 0, 1, 1], "modes": ["car"]})

    manifest_path = cache.folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "osm-overpass"
    assert manifest["tag"] == "test"
    assert manifest["bbox"] == [0, 0, 1, 1]
    assert manifest["modes"] == ["car"]
    assert "fetched_at" in manifest
    expected_sha = hashlib.sha256(written.read_bytes()).hexdigest()
    assert manifest["sha256"]["payload.parquet"] == expected_sha


def test_tag_is_slugified(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "São Paulo, Brazil!")
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    cache.write_geoparquet("payload", gdf)
    assert cache.folder.exists()
    folder_name = cache.folder.name
    for ch in folder_name:
        assert ch.isalnum() or ch in "-_.T"


def test_relative_path_uses_forward_slashes(tmp_path):
    cache = DownloadCache(tmp_path, "overture-cloud", "bbox_0_0_1_1")
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    cache.write_geoparquet("connectors", gdf)
    assert cache.relative_path is not None
    assert "\\" not in cache.relative_path
