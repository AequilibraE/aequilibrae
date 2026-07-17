"""Tests for ``StagedNetwork`` — schema invariants & MultiDiGraph round-trip."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer import StagedNetwork
from aequilibrae.project.network.importer.exceptions import StagedNetworkValidationError


def _make_minimal_staged():
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [100000, 100001, 100002],
            "geometry": [Point(0, 0), Point(0, 1), Point(1, 1)],
            "modes": ["c", "c", "c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1, 2],
            "a_node": [100000, 100001],
            "b_node": [100001, 100002],
            "direction": [0, 1],
            "modes": ["c", "c"],
            "link_type": ["residential", "primary"],
            "distance": [111000.0, 111000.0],
            "geometry": [LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    return StagedNetwork(nodes=nodes, links=links)


def test_minimal_staged_validates():
    net = _make_minimal_staged()
    net.validate()


def test_missing_required_node_column_raises():
    net = _make_minimal_staged()
    net.nodes = net.nodes.drop(columns=["modes"])
    with pytest.raises(StagedNetworkValidationError, match="modes"):
        net.validate()


def test_missing_required_link_column_raises():
    net = _make_minimal_staged()
    net.links = net.links.drop(columns=["distance"])
    with pytest.raises(StagedNetworkValidationError, match="distance"):
        net.validate()


def test_dangling_a_node_raises():
    net = _make_minimal_staged()
    net.links.loc[0, "a_node"] = 99999
    with pytest.raises(StagedNetworkValidationError, match="a_node"):
        net.validate()


def test_negative_distance_raises():
    net = _make_minimal_staged()
    net.links.loc[0, "distance"] = -1.0
    with pytest.raises(StagedNetworkValidationError, match="distance"):
        net.validate()


def test_invalid_direction_raises():
    net = _make_minimal_staged()
    net.links.loc[0, "direction"] = 2
    with pytest.raises(StagedNetworkValidationError, match="direction"):
        net.validate()


def test_node_id_below_floor_raises():
    net = _make_minimal_staged()
    net.nodes.loc[0, "node_id"] = 1
    with pytest.raises(StagedNetworkValidationError, match="10000"):
        net.validate()


def test_graph_round_trip_preserves_topology():
    net = _make_minimal_staged()
    g = net.to_graph()
    assert g.number_of_nodes() == 3
    # 2 links: one bidirectional (2 directed edges), one one-way (1 directed edge) = 3
    assert g.number_of_edges() == 3

    back = StagedNetwork.from_graph(g)
    assert set(back.nodes["node_id"]) == {100000, 100001, 100002}
    # The bidirectional link must be recombined into a single staged row, so
    # the reconstructed network has exactly the 2 original links, not 3.
    assert len(back.links) == 2
    assert back.links["link_id"].is_unique
    assert sorted(back.links["direction"]) == [0, 1]


def test_graph_round_trip_is_idempotent():
    net = _make_minimal_staged()
    once = StagedNetwork.from_graph(net.to_graph())
    twice = StagedNetwork.from_graph(once.to_graph())
    assert len(once.links) == len(twice.links) == 2
    assert sorted(once.links["direction"]) == sorted(twice.links["direction"])
    assert twice.links["link_id"].is_unique


def test_from_graph_requires_source_ref():
    import networkx as nx

    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    g.add_node(100000, x=0.0, y=0.0)
    g.add_node(100001, x=0.0, y=1.0)
    g.add_edge(100000, 100001, key=1, geometry=LineString([(0, 0), (0, 1)]))

    with pytest.raises(StagedNetworkValidationError, match="_source_ref"):
        StagedNetwork.from_graph(g)


def test_non_wgs84_crs_raises():
    net = _make_minimal_staged()
    net.nodes = net.nodes.to_crs("EPSG:3857")
    with pytest.raises(StagedNetworkValidationError, match="EPSG:4326"):
        net.validate()
