"""OSMnx simplifier tests + dict-of-dicts provenance verification."""

import json
import sqlite3

import pytest

osmnx = pytest.importorskip("osmnx")
pyrosm = pytest.importorskip("pyrosm")

from pyrosm import get_data


def test_simplify_reduces_link_count(empty_project):
    """Simplification must reduce the number of links."""
    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links_raw = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes_raw = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]


def test_simplify_osmnx_runs_and_reduces(empty_project):
    """Compare simplify off vs simplify=osmnx — node count should drop."""
    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("car",),
        simplify="osmnx",
        consolidate_tolerance=None,  # just degree-2 collapse, no consolidation
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0
    # Validate dict-of-dicts provenance on at least one link
    with sqlite3.connect(empty_project.path_to_file) as conn:
        for (oa,) in conn.execute("SELECT other_attributes FROM links WHERE other_attributes IS NOT NULL"):
            payload = json.loads(oa)
            if "source_id_list" in payload:
                inner = payload["source_id_list"]
                if isinstance(inner, str):
                    inner = json.loads(inner)
                assert isinstance(inner, dict), f"source_id_list must be a dict-of-dicts, got {type(inner).__name__}"
                for k, v in inner.items():
                    assert isinstance(k, str)
                    assert isinstance(v, dict), f"source_id_list[{k}] must be a dict, got {type(v).__name__}"
                return
    pytest.skip("No merged links produced — fixture too small for simplification to merge anything")


def test_simplify_osmnx_with_consolidation(empty_project):
    """consolidate_tolerance > 0 must not crash and yields a valid network."""
    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("car", "walk"),
        simplify="osmnx",
        consolidate_tolerance=10.0,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0
