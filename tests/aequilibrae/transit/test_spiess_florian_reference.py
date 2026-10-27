"""Spiess & Florian (1989) example network, imported as GTFS and built with ``TransitGraphBuilder``.

Spiess, H. and Florian, M. (1989). Optimal strategies: A new assignment model for transit networks.
Transportation Research Part B 23(2), 83-102.

Four lines serve stops A, X, Y and B. A strategy from A to B boards whichever of lines 1 and 2 comes
first. Line 2 riders stay on at X, then at Y take whichever of lines 3 and 4 comes first. Waits are
half a headway, and boarding, alighting and dwell are free, as in the paper. Expected values that
today's code gets wrong are strict ``xfail`` tests, so each repair is checked against the paper.
"""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.geo_utils import haversine

from .synthetic_gtfs import SERVICE_DATE, add_stop, add_timetable, base_tables, write_feed

LAT = -27.5
STOPS = {"A": 153.00, "X": 153.03, "Y": 153.06, "B": 153.09}
# Stops served, run time of each segment and headway, all in minutes
LINES = {
    "1": (["A", "B"], [25], 12),
    "2": (["A", "X", "Y"], [7, 6], 12),
    "3": (["X", "Y", "B"], [4, 4], 30),
    "4": (["Y", "B"], [10], 6),
}
# 07:00 to 08:00, inside all-day service, so every headway divides the period
AM_PEAK = 2

# Volumes of the strategy for one trip from A to B, keyed by edge type, line and stop. On-board edges
# use the stop where the segment starts, and dwell edges the stop where the vehicle dwells.
EXPECTED_VOLUMES = {
    ("boarding", "1", "A"): 1 / 2,
    ("boarding", "2", "A"): 1 / 2,
    ("boarding", "2", "X"): 0.0,
    ("boarding", "3", "X"): 0.0,
    ("boarding", "3", "Y"): 1 / 12,
    ("boarding", "4", "Y"): 5 / 12,
    ("on-board", "1", "A"): 1 / 2,
    ("on-board", "2", "A"): 1 / 2,
    ("on-board", "2", "X"): 1 / 2,
    ("on-board", "3", "X"): 0.0,
    ("on-board", "3", "Y"): 1 / 12,
    ("on-board", "4", "Y"): 5 / 12,
    ("dwell", "2", "X"): 1 / 2,
    ("dwell", "3", "Y"): 0.0,
    ("alighting", "1", "B"): 1 / 2,
    ("alighting", "2", "X"): 0.0,
    ("alighting", "2", "Y"): 1 / 2,
    ("alighting", "3", "Y"): 0.0,
    ("alighting", "3", "B"): 1 / 12,
    ("alighting", "4", "B"): 5 / 12,
}

SKIMS = ["trav_time", "boardings", "alightings", "in_vehicle_trav_time", "waiting_time", "transfers"]

# Boarding, alighting and dwell edges each add up to a few microseconds, and connectors a few millimetres
TIME_TOLERANCE = 0.01


@pytest.fixture
def sf_project(empty_project, tmp_path):
    tables = base_tables("SF")
    for stop, lon in STOPS.items():
        add_stop(tables, stop, LAT, lon)
    for route, (stops, run_minutes, headway) in LINES.items():
        add_timetable(tables, route, stops, run_minutes, headway, first=5 * 3600, last=11 * 3600)
    feed = write_feed(tmp_path / "spiess_florian.zip", tables)

    # One small zone centred on each of A and B
    for zone_id, stop in ((1, "A"), (2, "B")):
        lon = STOPS[stop]
        empty_project.network.zones.insert(
            zone_id=zone_id, geometry=MultiPolygon([box(lon - 0.002, LAT - 0.002, lon + 0.002, LAT + 0.002)])
        )
    with empty_project.db_connection as conn:
        conn.execute(
            "INSERT INTO periods (period_id, period_start, period_end, period_description) VALUES (?, ?, ?, ?)",
            [AM_PEAK, 7 * 3600, 8 * 3600, "AM peak"],
        )

    gtfs = empty_project.scenario.create_transit_database().new_gtfs_builder(agency="SF", file_path=str(feed))
    gtfs.load_date(SERVICE_DATE)
    gtfs.save_to_disk()
    return empty_project


def reference_builder(project) -> TransitGraphBuilder:
    builder = TransitGraphBuilder(project, period_id=AM_PEAK, with_walking_edges=False)
    # Free boarding, alighting and dwell, and an expected wait of half a headway, as in the paper
    builder.uniform_dwell_time = 1.0e-6
    builder.alighting_penalty = 0.0
    builder.wait_time_factor = 0.5
    builder.create_graph()
    return builder


def assign(builder: TransitGraphBuilder):
    graph = builder.to_transit_graph()
    demand = np.zeros((len(graph.centroids), len(graph.centroids)))
    demand[0, 1] = 1.0  # one trip from the zone at A to the zone at B
    matrix = AequilibraeMatrix(index=graph.centroids, index_name="zones")
    matrix["pt"] = demand

    transit_class = TransitClass(name="pt", graph=graph, matrix=matrix)
    assignment = TransitAssignment()
    assignment.add_class(transit_class)
    assignment.set_time_field("trav_time")
    assignment.set_frequency_field("freq")
    assignment.set_skimming_fields(SKIMS)
    assignment.set_algorithm("os")
    assignment.execute()

    edges = builder.edges.set_index("link_id")
    edges["volume"] = transit_class.results.get_load_results().volume
    return edges, assignment.get_skim_results()["pt"]


def gtfs_stops(project) -> dict[str, str]:
    """Map internal stop IDs to the stop IDs of the feed."""
    with project.transit_connection as conn:
        return dict(conn.execute("SELECT CAST(stop_id AS TEXT), stop FROM stops").fetchall())


def line_volumes(project, edges: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    stops = gtfs_stops(project)
    edges = edges[edges.link_type.isin(["boarding", "on-board", "dwell", "alighting"])]
    volumes = {}
    for edge in edges.itertuples():
        route = edge.line_id.split("_")[0]
        stop = LINES[route][0][edge.line_seg_idx] if edge.link_type == "on-board" else stops[edge.stop_id]
        volumes[edge.link_type, route, stop] = edge.volume
    return volumes


def test_strategy_volumes_match_published_example(sf_project):
    """Test that the optimal strategy loads every line segment, boarding and alighting as in the paper."""
    edges, _ = assign(reference_builder(sf_project))
    volumes = line_volumes(sf_project, edges)
    assert volumes.keys() == EXPECTED_VOLUMES.keys()
    for key, expected in EXPECTED_VOLUMES.items():
        assert volumes[key] == pytest.approx(expected, abs=1e-9), key


def test_expected_travel_time_matches_published_example(sf_project):
    """Test that the expected travel time from A to B is the published 27.75 minutes."""
    _, skims = assign(reference_builder(sf_project))
    assert skims["trav_time"][0, 1] == pytest.approx(27.75 * 60, abs=TIME_TOLERANCE)


def test_time_components_add_up_to_expected_travel_time(sf_project):
    """Test that in-vehicle time and waiting time match the strategy and add up to the travel time."""
    _, skims = assign(reference_builder(sf_project))
    # Half ride line 1, half ride line 2 to Y and then line 3 (1 in 6) or line 4 (5 in 6)
    in_vehicle = 0.5 * 25 + 0.5 * (7 + 6) + 1 / 12 * 4 + 5 / 12 * 10
    # A wait at A for either of lines 1 and 2, then half wait at Y for either of lines 3 and 4
    waiting = 1 / (2 / 12 + 2 / 12) + 0.5 / (2 / 30 + 2 / 6)
    assert skims["in_vehicle_trav_time"][0, 1] == pytest.approx(in_vehicle * 60, abs=TIME_TOLERANCE)
    assert skims["waiting_time"][0, 1] == pytest.approx(waiting * 60, abs=TIME_TOLERANCE)
    total = skims["in_vehicle_trav_time"][0, 1] + skims["waiting_time"][0, 1]
    assert total == pytest.approx(skims["trav_time"][0, 1], abs=TIME_TOLERANCE)


def test_boardings_count_the_transfer_at_y(sf_project):
    """Test that one trip boards and alights 1.5 times, since half the riders transfer at Y."""
    _, skims = assign(reference_builder(sf_project))
    assert skims["boardings"][0, 1] == pytest.approx(1.5)
    assert skims["alightings"][0, 1] == pytest.approx(1.5)


@pytest.mark.xfail(strict=True, reason="the transfers skim counts transfer edges, not alighting and boarding again")
def test_transfers_count_alighting_and_boarding_again(sf_project):
    """Test that half the trips from A to B transfer, as in the published strategy."""
    _, skims = assign(reference_builder(sf_project))
    assert skims["transfers"][0, 1] == pytest.approx(0.5)


@pytest.mark.xfail(strict=True, reason="skim buffers start at 0.0, so pairs with no strategy report 0.0")
def test_pairs_without_a_strategy_skim_as_infinite(sf_project):
    """Test that a pair with no transit strategy, B to A, skims as infinite rather than zero."""
    _, skims = assign(reference_builder(sf_project))
    assert np.isinf(skims["trav_time"][1, 0])


def test_assignment_conserves_flow(sf_project):
    """Test that demand is loaded in full, flow is conserved at every vertex and boardings equal alightings."""
    builder = reference_builder(sf_project)
    edges, _ = assign(builder)

    # Conservation holds whichever of a_node and b_node is the tail
    vertices = builder.vertices.set_index("node_id")
    count = vertices.index.max() + 1
    inflow = np.bincount(edges.a_node, weights=edges.volume, minlength=count)
    outflow = np.bincount(edges.b_node, weights=edges.volume, minlength=count)
    inner = vertices.index[~vertices.node_type.isin(["origin", "destination"])].to_numpy()
    np.testing.assert_allclose(inflow[inner], outflow[inner], atol=1e-12)

    assert edges.loc[edges.link_type == "access_connector", "volume"].sum() == pytest.approx(1.0)
    assert edges.loc[edges.link_type == "egress_connector", "volume"].sum() == pytest.approx(1.0)

    boardings = edges[edges.link_type == "boarding"].groupby("line_id").volume.sum()
    alightings = edges[edges.link_type == "alighting"].groupby("line_id").volume.sum()
    pd.testing.assert_series_equal(boardings, alightings, check_names=False)


def test_saved_graph_reloads_and_rebuilds_identically(sf_project):
    """Test that a graph built again, or saved and loaded, has identical vertices, edges and results."""
    builder = reference_builder(sf_project)
    builder.save()
    loaded = TransitGraphBuilder.from_db(sf_project, AM_PEAK)
    rebuilt = reference_builder(sf_project)

    def ordered(frame, key):
        return frame.drop(columns="geometry", errors="ignore").sort_values(key).reset_index(drop=True)

    for other in (loaded, rebuilt):
        pd.testing.assert_frame_equal(ordered(other.vertices, "node_id"), ordered(builder.vertices, "node_id"))
        pd.testing.assert_frame_equal(ordered(other.edges, "link_id"), ordered(builder.edges, "link_id"))
        pd.testing.assert_frame_equal(other.od_node_mapping, builder.od_node_mapping)

    edges, skims = assign(builder)
    for other in (loaded, rebuilt):
        other_edges, other_skims = assign(other)
        np.testing.assert_array_equal(other_edges.volume.to_numpy(), edges.volume.to_numpy())
        for name in SKIMS:
            np.testing.assert_array_equal(other_skims[name], skims[name])


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="walking edges are on by default but need parent stations"
)
def test_default_builder_builds_a_feed_without_parent_stations(sf_project):
    """Test that the builder's default settings build a graph from a feed that has no parent stations."""
    builder = TransitGraphBuilder(sf_project, period_id=AM_PEAK)
    builder.create_graph()


@pytest.mark.xfail(strict=True, reason="connectors are measured in EPSG:3857, which stretches distances by 1/cos(lat)")
def test_connectors_are_measured_geodesically(sf_project):
    """Test that a connector's walking time is the geodesic distance from the zone centroid to the stop."""
    builder = reference_builder(sf_project)
    stops = gtfs_stops(sf_project)
    access = builder.edges[(builder.edges.link_type == "access_connector") & (builder.edges.stop_id.map(stops) == "X")]
    # X is nearer to the zone at A than to the zone at B
    distance = haversine(STOPS["A"], LAT, STOPS["X"], LAT)
    assert access.trav_time.item() == pytest.approx(distance / builder.walking_speed, rel=1e-3)
