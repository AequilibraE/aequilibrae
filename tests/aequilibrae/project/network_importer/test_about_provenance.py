import sqlite3
import time

import pytest


def _import(empty_project, pbf_path):
    empty_project.network.importer.osm(pbf_path=pbf_path, modes=("car",), simplify=False)


def _about(path):
    with sqlite3.connect(path) as conn:
        return {
            r[0]: r[1]
            for r in conn.execute("SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'")
        }


def test_about_keys_populated_for_osm_pbf_source(empty_project, pbf_path):
    _import(empty_project, pbf_path)
    about = _about(empty_project.path_to_file)
    assert about["network_source"] == "osm"
    assert about["network_source_backend"] == "pyrosm"
    assert about["network_source_simplify"] == "false"
    assert about["network_source_modes"]
    assert about["network_source_fetched_at"]
    assert about["network_source_aequilibrae_version"]
    assert about["network_source_download_cache"] == ""
    assert "test.osm.pbf" in about["network_source_url"]


def test_about_keys_updated_in_place_on_reimport(empty_project, pbf_path):
    _import(empty_project, pbf_path)
    first_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]

    with empty_project.db_connection as conn:
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM link_types WHERE link_type NOT IN ('centroid_connector','default')")

    time.sleep(0.05)
    _import(empty_project, pbf_path)
    second_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]
    assert first_ts != second_ts

def test_about_not_written_when_db_write_fails(empty_project, pbf_path, monkeypatch):
    from aequilibrae.project.network.importer import db_writer

    def _boom(self, net):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(db_writer.SpatialiteWriter, "write", _boom)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        _import(empty_project, pbf_path)
    assert _about(empty_project.path_to_file) == {}
