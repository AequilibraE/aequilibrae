import sqlite3


def test_osm_import_preserves_all_link_types_for_active_modes(empty_project, pbf_path):
    empty_project.network.importer.osm(
        pbf_path=pbf_path,
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {r[0] for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()}
    pedestrian = {"footway", "pedestrian", "path", "steps", "cycleway"}
    assert link_types & pedestrian
