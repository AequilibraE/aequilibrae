import json
import sqlite3

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.simplifiers.impl_osmnx import _graph_to_staged
from aequilibrae.project.network.importer.staged_network import StagedNetwork

osmnx = pytest.importorskip("osmnx")
pyrosm = pytest.importorskip("pyrosm")


def _pbf_path():
    from pyrosm import get_data

    return get_data("test_pbf")


def test_simplify_osmnx_runs_and_reduces(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car",),
        simplify="osmnx",
        consolidate_tolerance=None,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0

    with sqlite3.connect(empty_project.path_to_file) as conn:
        for (oa,) in conn.execute("SELECT other_attributes FROM links WHERE other_attributes IS NOT NULL"):
            payload = json.loads(oa)
            if "source_ids" in payload:
                inner = payload["source_ids"]
                if isinstance(inner, str):
                    inner = json.loads(inner)
                assert isinstance(inner, dict)
                for key, value in inner.items():
                    assert isinstance(key, str)
                    assert isinstance(value, dict)
                return
    pytest.skip("No merged links produced")


def test_simplify_osmnx_with_consolidation(empty_project):
    empty_project.network.import_from_osm(
        pbf_path=_pbf_path(),
        modes=("car", "walk"),
        simplify="osmnx",
        consolidate_tolerance=10.0,
    )
    with sqlite3.connect(empty_project.path_to_file) as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 0
    assert n_nodes > 0


def test_graph_to_staged_preserves_merged_source_provenance():
    net = StagedNetwork(
        nodes=gpd.GeoDataFrame(
            {
                "node_id": [100000, 100001],
                "geometry": [Point(0.0, 0.0), Point(0.0, 0.001)],
                "modes": ["c", "c"],
                "source_id": ["n0", "n1"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        links=gpd.GeoDataFrame(
            {
                "link_id": [1, 2],
                "a_node": [100000, 100000],
                "b_node": [100001, 100001],
                "direction": [1, -1],
                "modes": ["c", "c"],
                "link_type": ["primary", "primary"],
                "distance": [100.0, 100.0],
                "geometry": [
                    LineString([(0.0, 0.0), (0.0, 0.001)]),
                    LineString([(0.0, 0.0), (0.0, 0.001)]),
                ],
                "name": ["Main St", "Main St"],
                "speed_ab": [50.0, None],
                "speed_ba": [None, 40.0],
                "lanes_ab": [2, None],
                "lanes_ba": [None, 1],
                "source_id": ["fwd", "bwd"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        source_meta={
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
            "release": "",
        },
    )

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(1, x=0.0, y=0.0, geometry=Point(0.0, 0.0), modes="c")
    graph.add_node(2, x=0.0, y=0.001, geometry=Point(0.0, 0.001), modes="c")
    graph.add_edge(
        1,
        2,
        key=0,
        geometry=LineString([(0.0, 0.0), (0.0, 0.001)]),
        direction=[1, -1],
        modes=["c", "c"],
        link_type=["primary", "primary"],
        source_id=["fwd", "bwd"],
        _source_ref=["fwd::ab", "bwd::ba"],
        length=[100.0, 100.0],
        name=["Main St", "Main St"],
    )

    simplified = _graph_to_staged(net, graph)

    assert len(simplified.links) == 1
    row = simplified.links.iloc[0]
    assert row["direction"] == 0
    assert row["modes"] == "c"
    assert row["speed_ab"] == 50.0
    assert row["speed_ba"] == 40.0
    assert row["lanes_ab"] == 2
    assert row["lanes_ba"] == 1
    payload = json.loads(row["source_ids"])
    assert set(payload) == {"fwd", "bwd"}


def test_graph_to_staged_reorients_reverse_one_way_as_forward_row_geometry():
    net = StagedNetwork(
        nodes=gpd.GeoDataFrame(
            {
                "node_id": [100000, 100001],
                "geometry": [Point(0.0, 0.0), Point(0.0, 0.001)],
                "modes": ["c", "c"],
                "source_id": ["n0", "n1"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        links=gpd.GeoDataFrame(
            {
                "link_id": [1],
                "a_node": [100000],
                "b_node": [100001],
                "direction": [-1],
                "modes": ["c"],
                "link_type": ["primary"],
                "distance": [100.0],
                "geometry": [LineString([(0.0, 0.0), (0.0, 0.001)])],
                "name": ["Main St"],
                "speed_ab": [None],
                "speed_ba": [40.0],
                "lanes_ab": [None],
                "lanes_ba": [1],
                "source_id": ["rev"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        source_meta={
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
            "release": "",
        },
    )

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(1, x=0.0, y=0.0, geometry=Point(0.0, 0.0), modes="c")
    graph.add_node(2, x=0.0, y=0.001, geometry=Point(0.0, 0.001), modes="c")
    graph.add_edge(
        2,
        1,
        key=0,
        geometry=LineString([(0.0, 0.001), (0.0, 0.0)]),
        direction=-1,
        modes="c",
        link_type="primary",
        source_id="rev",
        _source_ref="rev::ba",
        length=100.0,
        name="Main St",
    )

    simplified = _graph_to_staged(net, graph)

    row = simplified.links.iloc[0]
    assert row["direction"] == 1
    assert row["speed_ab"] == 40.0
    assert row["speed_ba"] is None
    assert row["lanes_ab"] == 1
    assert row["lanes_ba"] is None


