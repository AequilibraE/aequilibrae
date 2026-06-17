"""Tests for ``FileSource`` — round-trip user files via geopandas."""

import sqlite3
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point


def _fixtures():
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001, 10002],
            "geometry": [Point(0, 0), Point(0, 1), Point(1, 1)],
            "modes": ["c", "c", "c"],
            "custom_attr": ["a", "b", "c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1, 2],
            "a_node": [10000, 10001],
            "b_node": [10001, 10002],
            "direction": [0, 0],
            "modes": ["c", "c"],
            "link_type": ["residential", "primary"],
            "distance": [111000.0, 111000.0],
            "name": ["A", "B"],
            "surface": ["asphalt", "gravel"],
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 1), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    return nodes, links


def _assert_imported(project):
    with sqlite3.connect(project.path_to_file) as conn:
        assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM nodes").fetchone()[0] == 3


def test_file_source_geopackage(empty_project, tmp_path):
    nodes, links = _fixtures()
    nodes_path = tmp_path / "nodes.gpkg"
    links_path = tmp_path / "links.gpkg"
    nodes.to_file(nodes_path, driver="GPKG")
    links.to_file(links_path, driver="GPKG")

    empty_project.network.import_from_file(
        links_path=links_path,
        nodes_path=nodes_path,
        simplify=False,
    )
    _assert_imported(empty_project)


def test_file_source_geojson(empty_project, tmp_path):
    nodes, links = _fixtures()
    nodes_path = tmp_path / "nodes.geojson"
    links_path = tmp_path / "links.geojson"
    nodes.to_file(nodes_path, driver="GeoJSON")
    links.to_file(links_path, driver="GeoJSON")

    empty_project.network.import_from_file(
        links_path=links_path,
        nodes_path=nodes_path,
        simplify=False,
    )
    _assert_imported(empty_project)


def test_file_source_writes_no_download_cache(empty_project, tmp_path):
    nodes, links = _fixtures()
    nodes_path = tmp_path / "nodes.geojson"
    links_path = tmp_path / "links.geojson"
    nodes.to_file(nodes_path, driver="GeoJSON")
    links.to_file(links_path, driver="GeoJSON")

    empty_project.network.import_from_file(
        links_path=links_path,
        nodes_path=nodes_path,
        simplify=False,
    )
    cache = Path(empty_project.project_base_path) / "downloaded data"
    assert not cache.exists(), "FileSource must not write to downloaded data/"
