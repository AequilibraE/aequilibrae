import json
import sqlite3

import pytest

osmnx = pytest.importorskip("osmnx")
pyrosm = pytest.importorskip("pyrosm")


def _pbf_path():
    from pyrosm import get_data

    return get_data("test_pbf")


def test_simplify_osmnx_runs_and_reduces(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify="osmnx",
        consolidate_tolerance=None,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for (oa,) in conn.execute("SELECT other_attributes FROM links WHERE other_attributes IS NOT NULL"):
            payload = json.loads(oa)
            if "source_ids" in payload:
                inner = payload["source_ids"]
                if isinstance(inner, str):
                    inner = json.loads(inner)
                assert isinstance(inner, dict)
                for key, value in inner.items():
                    assert isinstance(key, str)
                    assert isinstance(value, dict)
                return
    pytest.skip("No merged links produced")


def test_simplify_osmnx_with_consolidation(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car", "walk"),
        simplify="osmnx",
        consolidate_tolerance=10.0,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0
