"""Direct tests for the ``DownloadCache`` helper (plan §4.6)."""

import gzip
import hashlib
import json

import pytest

from aequilibrae.project.network.importer import DownloadCache


def test_cache_is_lazy(tmp_path):
    """No folder is created unless something is written."""
    cache = DownloadCache(tmp_path, "osm-overpass", "vatican")
    assert cache.relative_path is None
    assert not (tmp_path / "downloaded data").exists()


def test_write_bytes_creates_folder(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "vatican")
    path = cache.write_bytes("response.json", b'{"hello": "world"}')
    assert path.exists()
    assert path.read_bytes() == b'{"hello": "world"}'
    assert cache.relative_path is not None
    assert "downloaded data" in cache.relative_path
    assert "osm-overpass" in cache.relative_path
    assert "vatican" in cache.relative_path


def test_large_payloads_are_gzipped(tmp_path):
    """Payloads > 10 MB land as .gz."""
    cache = DownloadCache(tmp_path, "osm-overpass", "huge")
    payload = b"x" * (11 * 1024 * 1024)
    path = cache.write_bytes("response.json", payload)
    assert path.suffix == ".gz"
    with gzip.open(path, "rb") as fh:
        assert fh.read() == payload


def test_manifest_includes_sha256(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "test")
    payload = b'{"hello": "world"}'
    cache.write_bytes("response.json", payload)
    cache.write_manifest({"bbox": [0, 0, 1, 1], "modes": ["car"]})

    manifest_path = cache.folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "osm-overpass"
    assert manifest["tag"] == "test"
    assert manifest["bbox"] == [0, 0, 1, 1]
    assert manifest["modes"] == ["car"]
    assert "fetched_at" in manifest
    expected_sha = hashlib.sha256(payload).hexdigest()
    assert manifest["sha256"]["response.json"] == expected_sha


def test_tag_is_slugified(tmp_path):
    cache = DownloadCache(tmp_path, "osm-overpass", "São Paulo, Brazil!")
    cache.write_bytes("x", b"x")
    assert cache.folder.exists()
    # No special chars
    folder_name = cache.folder.name
    for ch in folder_name:
        assert ch.isalnum() or ch in "-_.T"  # T allowed (timestamp), other alnum / - _ .


def test_relative_path_uses_forward_slashes(tmp_path):
    cache = DownloadCache(tmp_path, "overture-cloud", "bbox_0_0_1_1")
    cache.write_bytes("connectors.parquet", b"PAR1...")
    assert cache.relative_path is not None
    assert "\\" not in cache.relative_path
