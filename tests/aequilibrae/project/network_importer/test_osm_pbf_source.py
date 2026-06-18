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

    assert n_links > 10
    assert n_nodes > 10


def test_pbf_writes_no_download_cache(empty_project, tmp_path):
    from pathlib import Path

    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )
    cache = Path(empty_project.project_base_path) / "downloaded data"
    assert not cache.exists()


def test_pbf_mode_filter_only_keeps_walk_links(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(conn.execute("SELECT modes FROM links"))
    assert rows
    for (modes,) in rows:
        assert "w" in modes
        assert "c" not in modes


def test_pbf_link_types_are_preserved_uncapped(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car", "transit", "bicycle", "walk"),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {r[0] for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()}
    assert len(link_types) >= 3


def test_pbf_unknown_tags_land_in_other_attributes(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        sql = "SELECT other_attributes FROM links WHERE other_attributes IS NOT NULL LIMIT 50"
        for (other_attributes,) in conn.execute(sql):
            payload = json.loads(other_attributes)
            if payload:
                assert isinstance(payload, dict)
                assert "source_id" in payload
                return
    pytest.fail("No link had a non-empty other_attributes JSON payload")


def test_pbf_contract_fields_are_valid(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car", "walk"),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        rows = list(conn.execute("SELECT direction, distance, modes FROM links"))
        assert rows
        assert {row[0] for row in rows}.issubset({-1, 0, 1})
        assert all(row[1] > 0 for row in rows)
        assert all(row[2] for row in rows)
        assert any(row[0] != 0 for row in rows)


def test_pbf_about_provenance(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify=False,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        about = {
            r[0]: r[1]
            for r in conn.execute("SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'")
        }
    assert about["network_source"] == "osm"
    assert about["network_source_backend"] == "pyrosm"
    assert "test.osm.pbf" in about["network_source_url"]
    assert about["network_source_download_cache"] == ""


def test_no_alter_table_during_osm_import(empty_project):
    with sqlite3.connect(empty_project.path_to_file) as conn:
        before = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('links','nodes')"
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
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('links','nodes')"
            )
        }
    assert before == after
