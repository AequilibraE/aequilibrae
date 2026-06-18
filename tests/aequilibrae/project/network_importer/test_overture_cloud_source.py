"""Unit tests for ``OvertureCloudSource`` using a mocked record_batch_reader.

The plan reserves a live (gated) integration test as well; here we exercise the
full code path with an offline fixture so the test runs deterministically.
"""

import json
import sqlite3
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("overturemaps")

from shapely.geometry import LineString, Point, box
from shapely import to_wkb


def _make_connectors_table():
    """Synthesise a tiny Overture connector pyarrow.Table fixture."""
    points = [Point(0.0, 0.0), Point(0.0, 0.001), Point(0.001, 0.001), Point(0.001, 0.0)]
    geoms = [to_wkb(p) for p in points]
    ids = ["conn-0", "conn-1", "conn-2", "conn-3"]
    return pa.table({"id": ids, "geometry": geoms})


def _make_segments_table():
    """Synthesise a tiny Overture segment pyarrow.Table fixture.

    Two segments:
      - 'seg-0' with two connectors (0 → 1), residential, no rules
      - 'seg-1' with three connectors (1 → 2 → 3), with an intermediate
        connector that the importer must split on, plus a maxspeed.
    """
    seg0 = LineString([(0.0, 0.0), (0.0, 0.001)])
    seg1 = LineString([(0.0, 0.001), (0.001, 0.001), (0.001, 0.0)])
    ids = ["seg-0", "seg-1"]
    geoms = [to_wkb(seg0), to_wkb(seg1)]
    subtypes = ["road", "road"]
    classes = ["residential", "primary"]
    connectors = [
        [
            {"connector_id": "conn-0", "at": 0.0},
            {"connector_id": "conn-1", "at": 1.0},
        ],
        [
            {"connector_id": "conn-1", "at": 0.0},
            {"connector_id": "conn-2", "at": 0.5},
            {"connector_id": "conn-3", "at": 1.0},
        ],
    ]
    speed_limits = [
        None,
        [{"max_speed": {"value": 50, "unit": "km/h"}, "between": None, "when": None}],
    ]
    access_restrictions = [None, None]
    return pa.table(
        {
            "id": ids,
            "geometry": geoms,
            "subtype": subtypes,
            "class": classes,
            "connectors": connectors,
            "speed_limits": speed_limits,
            "access_restrictions": access_restrictions,
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
    # seg-0 = 1 link; seg-1 splits at intermediate connector → 2 links; total = 3
    assert n_links == 3
    # 4 connectors
    assert n_nodes == 4


def test_overture_writes_raw_payload_to_download_cache(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    base = Path(empty_project.project_base_path) / "downloaded data" / "overture-cloud"
    assert base.exists(), f"Expected download cache under {base}"
    subdirs = list(base.iterdir())
    assert subdirs, "No subdir created under overture-cloud/"
    cache = subdirs[0]
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
    for sab, sba, _lt in rows:
        # 50 km/h on a bidirectional link → both sides 50
        assert sab == 50.0 or sba == 50.0


def test_overture_rule_arrays_land_in_other_attributes(empty_project, monkeypatch):
    _install_mock(monkeypatch)
    empty_project.network.import_from_overture(
        model_area=box(-0.0005, -0.0005, 0.0015, 0.0015),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        for (oa,) in conn.execute("SELECT other_attributes FROM links WHERE link_type='primary'"):
            payload = json.loads(oa)
            # speed_limits array preserved verbatim (per plan §1.3 rule 7)
            assert "speed_limits" in payload, f"speed_limits must be preserved; got keys {list(payload.keys())}"


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
    # download cache must be populated
    assert about["network_source_download_cache"].startswith("downloaded data/overture-cloud/")
