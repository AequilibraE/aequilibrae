"""Patterns that visit a stop, or a stop pair, more than once, built and assigned through TransitGraphBuilder.

The GTFS importer still drops repeated stop pairs, so that case is written into the database directly.
"""

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder

from .synthetic_gtfs import add_period, add_stop, add_timetable, add_zone, base_tables, import_feed, internal_stops

LAT = -27.5
AM_PEAK = 2


def build(project) -> TransitGraphBuilder:
    builder = TransitGraphBuilder(project, period_id=AM_PEAK, with_walking_edges=False)
    builder.create_graph()
    return builder


def assign(builder: TransitGraphBuilder, origin: int, destination: int) -> tuple[pd.DataFrame, dict]:
    graph = builder.to_transit_graph()
    zones = list(graph.centroids)
    demand = np.zeros((len(zones), len(zones)))
    demand[zones.index(origin), zones.index(destination)] = 1.0
    matrix = AequilibraeMatrix(index=graph.centroids)
    matrix["pt"] = demand

    transit_class = TransitClass("pt", graph, matrix)
    assignment = TransitAssignment()
    assignment.add_class(transit_class)
    assignment.set_skimming_fields(["boardings", "trav_time"])
    assignment.set_algorithm("os")
    assignment.execute()

    edges = builder.edges.set_index("link_id")
    edges["volume"] = transit_class.results.get_load_results().volume
    skims = transit_class.results.skims
    return edges, {name: skims[name][zones.index(origin), zones.index(destination)] for name in skims}


def assert_wiring(builder: TransitGraphBuilder, stop_sequence: list[str], stops: dict[str, str]):
    """Check that one line's vertices and edges follow its stop sequence, one segment at a time."""
    vertices = builder.vertices
    node_type = vertices.set_index("node_id").node_type
    vertex = vertices.set_index("node_id")
    edges = builder.edges
    segments = len(stop_sequence) - 1

    for kind in ("boarding", "alighting"):
        line_vertices = vertices[vertices.node_type == kind]
        assert sorted(line_vertices.line_seg_idx) == list(range(segments)), kind
        assert line_vertices.line_seg_idx.is_unique, kind

    for seq in range(segments):
        board = edges[(edges.link_type == "boarding") & (edges.line_seg_idx == seq)].squeeze()
        ride = edges[(edges.link_type == "on-board") & (edges.line_seg_idx == seq)].squeeze()
        alight = edges[(edges.link_type == "alighting") & (edges.line_seg_idx == seq)].squeeze()

        stop_nodes = vertices[vertices.node_type == "stop"].set_index("stop_id").node_id
        assert board.a_node == stop_nodes[stops[stop_sequence[seq]]]
        assert board.b_node == ride.a_node and node_type[ride.a_node] == "boarding"
        assert ride.b_node == alight.a_node and node_type[ride.b_node] == "alighting"
        assert alight.b_node == stop_nodes[stops[stop_sequence[seq + 1]]]
        assert vertex.line_seg_idx[ride.a_node] == vertex.line_seg_idx[ride.b_node] == seq

        if seq:
            dwell = edges[(edges.link_type == "dwell") & (edges.line_seg_idx == seq)].squeeze()
            previous = edges[(edges.link_type == "on-board") & (edges.line_seg_idx == seq - 1)].squeeze()
            assert dwell.a_node == previous.b_node
            assert dwell.b_node == ride.a_node
            assert dwell.stop_id == stops[stop_sequence[seq]]

    assert (edges.link_type == "dwell").sum() == segments - 1


def assert_conserved(builder: TransitGraphBuilder, edges: pd.DataFrame):
    vertices = builder.vertices.set_index("node_id")
    count = vertices.index.max() + 1
    outflow = np.bincount(edges.a_node, weights=edges.volume, minlength=count)
    inflow = np.bincount(edges.b_node, weights=edges.volume, minlength=count)
    inner = vertices.index[~vertices.node_type.isin(["origin", "destination"])].to_numpy()
    np.testing.assert_allclose(inflow[inner], outflow[inner], atol=1e-12)
    assert edges.loc[edges.link_type == "access_connector", "volume"].sum() == pytest.approx(1.0)
    assert edges.loc[edges.link_type == "egress_connector", "volume"].sum() == pytest.approx(1.0)


@pytest.fixture
def project(empty_project):
    add_period(empty_project, AM_PEAK, 7 * 3600, 8 * 3600)
    return empty_project


def test_lasso_visiting_a_stop_twice(project, tmp_path):
    """Test that a pattern passing stop B twice gets one vertex per segment and carries a trip around its loop."""
    tables = base_tables()
    for stop, lat, lon in [("A", 0.0, 0.00), ("B", 0.0, 0.01), ("C", 0.01, 0.02), ("D", 0.02, 0.01), ("E", 0.0, 0.05)]:
        add_stop(tables, stop, LAT + lat, 153.0 + lon)
    sequence = ["A", "B", "C", "D", "B", "E"]
    add_timetable(tables, "L", sequence, [3, 3, 3, 3, 3], 10, first=5 * 3600, last=11 * 3600)
    add_zone(project, 1, LAT, 153.0)
    add_zone(project, 2, LAT, 153.05)
    import_feed(project, tables, tmp_path / "lasso.zip")

    builder = build(project)
    assert_wiring(builder, sequence, internal_stops(project))

    edges, skims = assign(builder, 1, 2)
    assert_conserved(builder, edges)
    # Alighting at B to wait for the next vehicle costs a penalty, so the trip stays on around the loop
    assert skims["boardings"] == pytest.approx(1.0)
    np.testing.assert_allclose(edges.loc[edges.link_type.isin(["on-board", "dwell"]), "volume"], 1.0)


def test_pattern_repeating_a_stop_pair(project, tmp_path):
    """Test that a pattern running A to B twice keeps both segments, wired and loaded in order."""
    tables = base_tables()
    for stop, lon in [("A", 153.00), ("B", 153.01), ("C", 153.02), ("D", 153.03)]:
        add_stop(tables, stop, LAT, lon)
    add_timetable(tables, "R", ["A", "B", "C", "D"], [3, 3, 3], 10, first=5 * 3600, last=11 * 3600)
    add_zone(project, 1, LAT, 153.0)
    add_zone(project, 2, LAT, 153.01)
    import_feed(project, tables, tmp_path / "pairs.zip")

    # The importer would drop the repeated pair, so the pattern becomes A, B, A, B in the database
    stops = internal_stops(project)
    sequence = ["A", "B", "A", "B"]
    with project.transit_connection as conn:
        for seq in range(3):
            conn.execute(
                "UPDATE route_links SET from_stop = ?, to_stop = ? WHERE seq = ?",
                [int(stops[sequence[seq]]), int(stops[sequence[seq + 1]]), seq],
            )

    builder = build(project)
    assert_wiring(builder, sequence, stops)
    assert set(builder.vertices[builder.vertices.node_type == "stop"].stop_id) == {stops["A"], stops["B"]}

    edges, skims = assign(builder, 1, 2)
    assert_conserved(builder, edges)
    assert skims["boardings"] == pytest.approx(1.0)
