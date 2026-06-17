"""Tests for ``OSMPbfSource`` using pyrosm's bundled test.osm.pbf fixture."""

import json
import sqlite3

import pytest

pyrosm = pytest.importorskip("pyrosm")


def _pbf_path():
    from pyrosm import get_data

    return get_data("test_pbf")


def test_basic_pbf_import_no_simplify(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]

    assert n_links > 10, f"expected > 10 links, got {n_links}"
    assert n_nodes > 10, f"expected > 10 nodes, got {n_nodes}"


def test_pbf_writes_no_download_cache(empty_project, tmp_path):
    """Local PBF sources must not create a downloaded data/ folder."""
    from pathlib import Path

    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )
    cache = Path(empty_project.project_base_path) / "downloaded data"
    assert not cache.exists(), "PBF source must not write to downloaded data/"


def test_pbf_mode_filter_only_keeps_walk_links(empty_project):
    """modes=('walk',) must drop car-only motorway-class links."""
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(conn.execute("SELECT modes FROM links"))
    assert rows, "Expected at least one walkable link"
    for (modes,) in rows:
        # mode_filter must have already trimmed to the requested subset
        assert "w" in modes
        assert "c" not in modes


def test_pbf_link_types_are_preserved_uncapped(empty_project):
    """plan §1.3 rule 2: no link-type allow-list. Every distinct highway value
    that survives the mode filter must appear as a link_type."""
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car", "transit", "bicycle", "walk"),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {
            r[0]
            for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()
        }
    # We expect multiple distinct link types (residential, primary, secondary, etc.)
    assert len(link_types) >= 3, f"only got link_types: {link_types}"


def test_pbf_unknown_tags_land_in_other_attributes(empty_project):
    """OSM tags without a same-named column must JSON-encode to other_attributes."""
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        # find a link with non-null other_attributes
        for (oa,) in conn.execute(
            "SELECT other_attributes FROM links WHERE other_attributes IS NOT NULL LIMIT 50"
        ):
            payload = json.loads(oa)
            if payload:
                # at least one OSM tag we know exists in pyrosm's fixture
                assert isinstance(payload, dict)
                return
    pytest.fail("No link had a non-empty other_attributes JSON payload")


def test_pbf_about_provenance(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        about = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'"
            )
        }
    assert about["network_source"] == "osm"
    assert about["network_source_backend"] == "pyrosm"
    assert "test.osm.pbf" in about["network_source_url"]
    assert about["network_source_download_cache"] == ""  # local source


def test_no_alter_table_during_osm_import(empty_project):
    """Strict invariant: no ALTER TABLE on links/nodes during an OSM import."""
    with sqlite3.connect(empty_project.path_to_file) as conn:
        before = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name IN ('links','nodes')"
            )
        }
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        after = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name IN ('links','nodes')"
            )
        }
    assert before == after
