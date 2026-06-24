import sqlite3

import pytest


def test_osm_import_preserves_all_link_types_for_active_modes(empty_project):
    pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {r[0] for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()}
    pedestrian = {"footway", "pedestrian", "path", "steps", "cycleway"}
    assert link_types & pedestrian
