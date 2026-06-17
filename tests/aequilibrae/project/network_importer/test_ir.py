"""Tests for ``RoutableNetwork`` (IR) — schema invariants & MultiDiGraph round-trip."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer import RoutableNetwork
from aequilibrae.project.network.importer.exceptions import IRValidationError


def _make_minimal_ir():
    nodes = gpd.GeoDataFrame(
        {
            "node_id": [10000, 10001, 10002],
            "geometry": [Point(0, 0), Point(0, 1), Point(1, 1)],
            "modes": ["c", "c", "c"],
        },
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1, 2],
            "a_node": [10000, 10001],
            "b_node": [10001, 10002],
            "direction": [0, 1],
            "modes": ["c", "c"],
            "link_type": ["residential", "primary"],
            "distance": [111000.0, 111000.0],
            "geometry": [LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    return RoutableNetwork(nodes=nodes, links=links)


def test_minimal_ir_validates():
    ir = _make_minimal_ir()
    ir.validate()  # should not raise


def test_missing_required_node_column_raises():
    ir = _make_minimal_ir()
    ir.nodes = ir.nodes.drop(columns=["modes"])
    with pytest.raises(IRValidationError, match="modes"):
        ir.validate()


def test_missing_required_link_column_raises():
    ir = _make_minimal_ir()
    ir.links = ir.links.drop(columns=["distance"])
    with pytest.raises(IRValidationError, match="distance"):
        ir.validate()


def test_dangling_a_node_raises():
    ir = _make_minimal_ir()
    ir.links.loc[0, "a_node"] = 99999
    with pytest.raises(IRValidationError, match="a_node"):
        ir.validate()


def test_negative_distance_raises():
    ir = _make_minimal_ir()
    ir.links.loc[0, "distance"] = -1.0
    with pytest.raises(IRValidationError, match="distance"):
        ir.validate()


def test_invalid_direction_raises():
    ir = _make_minimal_ir()
    ir.links.loc[0, "direction"] = 2
    with pytest.raises(IRValidationError, match="direction"):
        ir.validate()


def test_node_id_below_floor_raises():
    ir = _make_minimal_ir()
    ir.nodes.loc[0, "node_id"] = 1
    with pytest.raises(IRValidationError, match="10000"):
        ir.validate()


def test_multidigraph_round_trip_preserves_topology():
    ir = _make_minimal_ir()
    g = ir.to_multidigraph()
    assert g.number_of_nodes() == 3
    # 2 links: one bidirectional (2 directed edges), one one-way (1 directed edge) = 3
    assert g.number_of_edges() == 3

    back = RoutableNetwork.from_multidigraph(g)
    # Topology preserved
    assert set(back.nodes["node_id"]) == {10000, 10001, 10002}
    assert len(back.links) == 3  # split bidirectional → 2 + 1 one-way
