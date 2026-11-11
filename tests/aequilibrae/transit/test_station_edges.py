"""Walks between the stops of a parent station, as walking edges or outer transfers."""

import numpy as np
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass, TransitCosting
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.geo_utils import haversine

from .synthetic_gtfs import add_period, add_stop, add_timetable, add_zone, base_tables, import_feed, internal_stops

LAT = -27.5
# S1 and S2 share a station half way between the zones at A and B, about 50 metres apart
STOPS = {"A": 153.00, "S1": 153.049, "S2": 153.0495, "B": 153.10}
AM_PEAK = 2


@pytest.fixture
def project(empty_project, tmp_path):
    tables = base_tables()
    for stop, lon in STOPS.items():
        columns = {"parent_station": "ST"} if stop.startswith("S") else {}
        add_stop(tables, stop, LAT, lon, **columns)
    add_timetable(tables, "1", ["A", "S1"], [10], 10, first=5 * 3600, last=11 * 3600)
    add_timetable(tables, "2", ["S2", "B"], [10], 10, first=5 * 3600, last=11 * 3600)
    add_zone(empty_project, 1, LAT, STOPS["A"])
    add_zone(empty_project, 2, LAT, STOPS["B"])
    add_period(empty_project, AM_PEAK, 7 * 3600, 8 * 3600)
    import_feed(empty_project, tables, tmp_path / "station.zip")
    return empty_project


def station_edges(builder):
    return builder.edges[builder.edges.link_type.isin(["walking", "outer_transfer"])]


def build(project, **kwargs) -> TransitGraphBuilder:
    builder = TransitGraphBuilder(project, period_id=AM_PEAK, **kwargs)
    builder.create_graph()
    return builder


@pytest.mark.parametrize(
    "walking,outer,link_type",
    [(True, False, "walking"), (False, True, "outer_transfer"), (True, True, "outer_transfer"), (False, False, None)],
)
def test_station_stops_get_one_edge_each_way(project, walking, outer, link_type):
    """Test that each ordered pair of station stops gets one stop-to-stop walk, an outer transfer if requested."""
    builder = build(project, with_walking_edges=walking, with_outer_stop_transfers=outer)
    edges = station_edges(builder)
    if link_type is None:
        assert edges.empty
        return

    stops = internal_stops(project)
    stop_nodes = builder.vertices[builder.vertices.node_type == "stop"].set_index("stop_id").node_id
    s1, s2 = stop_nodes[stops["S1"]], stop_nodes[stops["S2"]]
    assert set(edges.link_type) == {link_type}
    assert sorted(zip(edges.a_node, edges.b_node, strict=True)) == sorted([(s1, s2), (s2, s1)])
    assert np.isinf(edges.line_freq).all()
    distance = haversine(STOPS["S1"], LAT, STOPS["S2"], LAT)
    np.testing.assert_allclose(edges.walk_distance, distance)
    np.testing.assert_allclose(edges.walk_time, distance / builder.walking_speed)
    assert sorted(edges.segment_id) == sorted([f"{stops['S1']}:{stops['S2']}", f"{stops['S2']}:{stops['S1']}"])


def test_transfers_walk_between_station_stops(project):
    """Test that a trip from A to B alights at S1, walks to S2 and boards again, counting one outer transfer."""
    builder = build(project, with_outer_stop_transfers=True)
    graph = builder.to_transit_graph()
    matrix = AequilibraeMatrix(index=graph.centroids)
    matrix["pt"] = np.array([[0.0, 1.0], [0.0, 0.0]])

    transit_class = TransitClass("pt", graph, matrix, costing=TransitCosting(alighting_penalty=0.0))
    assignment = TransitAssignment()
    assignment.add_class(transit_class)
    assignment.set_skimming_fields(["boardings", "transfers", "outer_transfers", "transfer_time"])
    assignment.set_algorithm("os")
    assignment.execute()

    skims = transit_class.results.skims
    assert skims["boardings"][0, 1] == pytest.approx(2.0)
    assert skims["transfers"][0, 1] == pytest.approx(1.0)
    assert skims["outer_transfers"][0, 1] == pytest.approx(1.0)
    assert skims["transfer_time"][0, 1] == pytest.approx(haversine(STOPS["S1"], LAT, STOPS["S2"], LAT))

    volumes = transit_class.results.get_load_results().volume
    walks = station_edges(builder).set_index("link_id")
    assert sorted(volumes.loc[walks.index]) == pytest.approx([0.0, 1.0])
