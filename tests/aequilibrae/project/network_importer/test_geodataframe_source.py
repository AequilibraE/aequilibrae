"""Tests for ``GeoDataFrameSource`` — the simplest source."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer import DownloadCache
from aequilibrae.project.network.importer.exceptions import StagedNetworkValidationError
from aequilibrae.project.network.importer.sources.generic.geodataframe import (
    GeoDataFrameSource,
)


def _nodes(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001],
            "geometry": [Point(0, 0), Point(0, 1)],
            "modes": ["c", "c"],
        },
        crs=crs,
    )


def _links(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
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
        crs=crs,
    )


def test_basic_acquire(tmp_path):
    cache = DownloadCache(tmp_path, "geodataframe", "test")
    src = GeoDataFrameSource(nodes=_nodes(), links=_links())
    net = src.acquire(modes=("car",), download_cache=cache)
    assert len(net.nodes) == 2
    assert len(net.links) == 1
    assert cache.relative_path is None  # nothing should have been written


def test_reprojects_to_4326(tmp_path):
    cache = DownloadCache(tmp_path, "geodataframe", "test")
    nodes = _nodes(crs="EPSG:3857")
    links = _links(crs="EPSG:3857")
    src = GeoDataFrameSource(nodes=nodes, links=links)
    net = src.acquire(modes=("car",), download_cache=cache)
    assert str(net.nodes.crs).upper() == "EPSG:4326"
    assert str(net.links.crs).upper() == "EPSG:4326"


def test_raises_when_crs_missing(tmp_path):
    cache = DownloadCache(tmp_path, "geodataframe", "test")
    nodes = _nodes(crs=None)
    links = _links(crs=None)
    # geopandas no-crs constructions: re-set to None
    nodes = nodes.set_crs(None, allow_override=True)
    links = links.set_crs(None, allow_override=True)
    src = GeoDataFrameSource(nodes=nodes, links=links)
    with pytest.raises(StagedNetworkValidationError, match="CRS"):
        src.acquire(modes=("car",), download_cache=cache)


def test_column_mapping(tmp_path):
    cache = DownloadCache(tmp_path, "geodataframe", "test")
    nodes = _nodes().rename(columns={"node_id": "nid"})
    links = _links().rename(columns={"link_id": "lid"})
    src = GeoDataFrameSource(
        nodes=nodes,
        links=links,
        column_mapping={"nid": "node_id", "lid": "link_id"},
    )
    net = src.acquire(modes=("car",), download_cache=cache)
    assert "node_id" in net.nodes.columns
    assert "link_id" in net.links.columns


def test_validation_rejects_dangling_a_node(tmp_path):
    cache = DownloadCache(tmp_path, "geodataframe", "test")
    nodes = _nodes()
    links = _links()
    links.loc[0, "a_node"] = 99999
    src = GeoDataFrameSource(nodes=nodes, links=links)
    with pytest.raises(StagedNetworkValidationError, match="a_node"):
        src.acquire(modes=("car",), download_cache=cache)
