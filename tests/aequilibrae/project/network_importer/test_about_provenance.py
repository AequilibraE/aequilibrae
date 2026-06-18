import sqlite3
import time

import pytest

pyrosm = pytest.importorskip("pyrosm")


def _pbf_path():
    from pyrosm import get_data

    return get_data("test_pbf")


def _import(empty_project):
    empty_project.network.import_from_osm(pbf_path=_pbf_path(), modes=("car",), simplify=False)


def _about(path):
    with sqlite3.connect(path) as conn:
        return {
            r[0]: r[1]
            for r in conn.execute("SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'")
        }


def test_about_keys_populated_for_osm_pbf_source(empty_project):
    _import(empty_project)
    about = _about(empty_project.path_to_file)
    assert about["network_source"] == "osm"
    assert about["network_source_backend"] == "pyrosm"
    assert about["network_source_simplify"] == "false"
    assert about["network_source_modes"]
    assert about["network_source_fetched_at"]
    assert about["network_source_aequilibrae_version"]
    assert about["network_source_download_cache"] == ""


def test_about_keys_updated_in_place_on_reimport(empty_project):
    _import(empty_project)
    first_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]

    with empty_project.db_connection as conn:
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM link_types WHERE link_type NOT IN ('centroid_connector','default')")

    time.sleep(0.05)
    _import(empty_project)
    second_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]
    assert first_ts != second_ts
