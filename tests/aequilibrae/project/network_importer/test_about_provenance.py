"""About-table provenance written by the importer (plan §10)."""

import sqlite3

import geopandas as gpd
from shapely.geometry import LineString, Point


def _import(empty_project):
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001],
            "geometry": [Point(0, 0), Point(0, 1)],
            "modes": ["c", "c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [10000],
            "b_node": [10001],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["residential"],
            "distance": [111000.0],
            "geometry": [LineString([(0, 0), (0, 1)])],
        },
        crs="EPSG:4326",
    )
    empty_project.network.import_from_geodataframes(nodes=nodes, links=links, simplify=False)


def _about(path):
    with sqlite3.connect(path) as conn:
        return {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT infoname, infovalue FROM about WHERE infoname LIKE 'network_source%'"
            )
        }


def test_about_keys_populated_for_geodataframe_source(empty_project):
    _import(empty_project)
    about = _about(empty_project.path_to_file)
    assert about["network_source"] == "geodataframe"
    assert about["network_source_backend"] == "user"
    assert about["network_source_simplify"] == "false"
    assert about["network_source_modes"]  # non-empty
    assert about["network_source_fetched_at"]
    assert about["network_source_aequilibrae_version"]
    # Download cache must be empty for local-data sources
    assert about["network_source_download_cache"] == ""


def test_about_keys_updated_in_place_on_reimport(empty_project):
    _import(empty_project)
    first_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]

    # Clear links so a re-import doesn't trip uniqueness
    with empty_project.db_connection as conn:
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM link_types WHERE link_type NOT IN ('centroid_connector','default')")

    import time
    time.sleep(0.05)
    _import(empty_project)
    second_ts = _about(empty_project.path_to_file)["network_source_fetched_at"]
    assert first_ts != second_ts
