import json
import sqlite3
from pathlib import Path

import pytest
from shapely import to_wkb
from shapely.geometry import LineString, Point, box

pa = pytest.importorskip("pyarrow")
pytest.importorskip("overturemaps")


def _make_connectors_table():
    points = [Point(0.0, 0.0), Point(0.0, 0.001), Point(0.001, 0.001), Point(0.001, 0.0)]
    return pa.table({"id": ["conn-0", "conn-1", "conn-2", "conn-3"], "geometry": [to_wkb(p) for p in points]})


def _make_segments_table():
    seg0 = LineString([(0.0, 0.0), (0.0, 0.001)])
    seg1 = LineString([(0.0, 0.001), (0.001, 0.001), (0.001, 0.0)])
    return pa.table(
        {
            "id": ["seg-0", "seg-1"],
            "geometry": [to_wkb(seg0), to_wkb(seg1)],
            "subtype": ["road", "road"],
            "class": ["residential", "primary"],
            "connectors": [
                [{"connector_id": "conn-0", "at": 0.0}, {"connector_id": "conn-1", "at": 1.0}],
                [
                    {"connector_id": "conn-1", "at": 0.0},
                    {"connector_id": "conn-2", "at": 0.5},
                    {"connector_id": "conn-3", "at": 1.0},
                ],
            ],
            "speed_limits": [None, [{"max_speed": {"value": 50, "unit": "km/h"}, "between": None, "when": None}]],
            "access_restrictions": [None, None],
        }
    )


class _FakeReader:
    def __init__(self, table):
        self._table = table

    def read_all(self):
        return self._table


def _install_mock(monkeypatch):
    import overturemaps

    def _fake_rbr(theme_type, bbox=None, **kwargs):
        if theme_type == "connector":
            return _FakeReader(_make_connectors_table())
        if theme_type == "segment":
            return _FakeReader(_make_segments_table())
        raise AssertionError(f"unexpected theme: {theme_type}")

    monkeypatch.setattr(overturemaps, "record_batch_reader", _fake_rbr)


def test_overture_cloud_import_splits_intermediate_connectors(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links == 3
    assert n_nodes == 4


def test_overture_writes_raw_payload_to_download_cache(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    base = Path(empty_project.project_base_path) / "downloaded data" / "overture-cloud"
    assert base.exists()
    cache = list(base.iterdir())[0]
    assert (cache / "segments.parquet").exists()
    assert (cache / "connectors.parquet").exists()
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "overture-cloud"
    assert manifest["segments_rows"] == 2
    assert manifest["connectors_rows"] == 4


def test_overture_speed_limit_parsed(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(conn.execute("SELECT speed_ab, speed_ba, link_type FROM links WHERE link_type='primary'"))
    assert rows
    for speed_ab, speed_ba, _link_type in rows:
        assert speed_ab == 50.0 or speed_ba == 50.0


def test_overture_rule_arrays_land_in_other_attributes(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        for (other_attributes,) in conn.execute("SELECT other_attributes FROM links WHERE link_type='primary'"):
            payload = json.loads(other_attributes)
            assert "speed_limits" in payload


def test_overture_about_provenance(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        about = {
            r[0]: r[1]
            for r in conn.execute("SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'")
        }
    assert about["network_source"] == "overture"
    assert about["network_source_backend"] == "cloud"
    assert "overturemaps-us-west-2" in about["network_source_url"]
    assert about["network_source_download_cache"].startswith("downloaded data/overture-cloud/")
