"""Trips, periods and the schedule attributes TransitGraphBuilder derives from them, on synthetic feeds.

Times count seconds from the start of the service day, so trips after midnight are stored above 86400.
"""

import logging

import pytest

from aequilibrae.paths import TransitCosting
from aequilibrae.transit.service_periods import period_window
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder

from .synthetic_gtfs import add_period, add_stop, add_trips, add_zone, base_tables, import_feed, internal_stops

LAT = -27.5
STOPS = {"P": 153.00, "Q": 153.01, "S": 153.02}
AM_PEAK = 2  # 07:00 to 08:00
HOUR = 3600


def hours(*times: float) -> list[int]:
    return [round(t * HOUR) for t in times]


@pytest.fixture
def project(empty_project):
    for zone_id, stop in ((1, "P"), (2, "S")):
        add_zone(empty_project, zone_id, LAT, STOPS[stop])
    add_period(empty_project, AM_PEAK, 7 * HOUR, 8 * HOUR)
    return empty_project


def feed(*routes):
    """Tables with stops P, Q and S and the given (route, stops, run minutes, departures) routes."""
    tables = base_tables()
    for stop, lon in STOPS.items():
        add_stop(tables, stop, LAT, lon)
    for route in routes:
        add_trips(tables, *route)
    return tables


def build(project, period_id=AM_PEAK, **kwargs) -> TransitGraphBuilder:
    builder = TransitGraphBuilder(project, period_id=period_id, with_walking_edges=False, **kwargs)
    builder.create_graph()
    return builder


def line_edges(builder: TransitGraphBuilder, route: str, link_type: str):
    edges = builder.edges[builder.edges.link_type == link_type]
    return edges[edges.line_id.str.split("_").str[0] == route].sort_values("line_seg_idx")


def test_default_period_spans_the_whole_service_day(project, tmp_path):
    """Test that the default period runs to 48:00:00 and keeps trips that leave after midnight."""
    import_feed(project, feed(("N", ["P", "Q"], [10], hours(23.5, 23.75, 24.25, 24.5))), tmp_path / "f.zip")
    with project.db_connection as conn:
        assert period_window(conn, 1) == (0, 48 * HOUR)

    boarding = line_edges(build(project, period_id=1), "N", "boarding").iloc[0]
    assert boarding.trip_count == 4
    # The whole-day period averages over one day, not two
    assert boarding.line_freq == pytest.approx(4 / (24 * HOUR))


def test_periods_after_midnight_select_their_trips(project, tmp_path):
    """Test that a period from 24:00:00 to 25:00:00 selects the trips leaving then."""
    import_feed(project, feed(("N", ["P", "Q"], [10], hours(23.5, 23.75, 24.25, 24.5))), tmp_path / "f.zip")
    add_period(project, 3, 24 * HOUR, 25 * HOUR)

    boarding = line_edges(build(project, period_id=3), "N", "boarding").iloc[0]
    assert boarding.trip_count == 2
    assert boarding.line_freq == pytest.approx(2 / HOUR)
    assert boarding.headway == pytest.approx(0.25 * HOUR)


@pytest.mark.parametrize("rule,trips", [("start", 2), ("end", 2), ("midpoint", 3), ("any", 4)])
def test_trip_inclusion_rules(project, tmp_path, rule, trips):
    """Test that each trip-level rule assigns the expected trips that run across the period's edges."""
    # Trips take 40 minutes: 06:40-07:20, 06:50-07:30, 07:30-08:10 and 07:50-08:30
    departures = hours(6 + 40 / 60, 6 + 50 / 60, 7.5, 7 + 50 / 60)
    import_feed(project, feed(("R", ["P", "Q", "S"], [20, 20], departures)), tmp_path / "f.zip")

    builder = build(project, trip_inclusion=rule)
    boardings = line_edges(builder, "R", "boarding")
    assert boardings.trip_count.tolist() == [trips, trips]
    assert boardings.line_freq.tolist() == [pytest.approx(trips / HOUR)] * 2
    # Every segment of a pattern belongs to the same trips, so the graph is never truncated
    assert len(line_edges(builder, "R", "on-board")) == 2


def test_invalid_trip_inclusion_rule_is_rejected(project, tmp_path):
    """Test that an unknown trip inclusion rule raises."""
    import_feed(project, feed(("R", ["P", "Q"], [20], hours(7.2))), tmp_path / "f.zip")
    with pytest.raises(ValueError, match="trip_inclusion"):
        TransitGraphBuilder(project, period_id=AM_PEAK, trip_inclusion="sometimes")


def test_duplicate_trips_add_capacity_but_not_frequency(project, tmp_path):
    """Test that two trips leaving together count once for frequency and twice for trips and capacity."""
    import_feed(
        project, feed(("D", ["P", "Q"], [10], hours(7 + 10 / 60, 7 + 10 / 60, 7 + 40 / 60))), tmp_path / "f.zip"
    )
    with project.transit_connection as conn:
        capacity = conn.execute("SELECT total_capacity FROM routes").fetchone()[0]

    builder = build(project)
    boarding = line_edges(builder, "D", "boarding").iloc[0]
    assert boarding.trip_count == 3
    assert boarding.line_freq == pytest.approx(2 / HOUR)
    assert boarding.headway == pytest.approx(0.5 * HOUR)
    assert line_edges(builder, "D", "on-board").capacity.item() == pytest.approx(3 * capacity)


def test_a_single_trip_runs_once_in_the_period(project, tmp_path):
    """Test that a pattern with one trip in the period gets the period length as its headway."""
    import_feed(project, feed(("O", ["P", "Q"], [10], hours(7 + 20 / 60))), tmp_path / "f.zip")
    boarding = line_edges(build(project), "O", "boarding").iloc[0]
    assert boarding.trip_count == 1
    assert boarding.headway == pytest.approx(HOUR)
    assert boarding.line_freq == pytest.approx(1 / HOUR)


def test_run_times_average_the_period_trips(project, tmp_path):
    """Test that run times are the mean over the period's trips, not over the whole day."""
    import_feed(
        project,
        feed(("T", ["P", "Q"], [10], hours(6.5, 7.25)), ("T2", ["P", "Q"], [20], hours(7.5))),
        tmp_path / "f.zip",
    )
    with project.transit_connection as conn:
        # Slow the 06:30 trip down, which the AM peak does not include
        conn.execute(
            "UPDATE trips_schedule SET arrival = arrival + 600, departure = departure + 600 WHERE seq = 1 "
            "AND trip_id = (SELECT MIN(trip_id) FROM trips_schedule)"
        )
    builder = build(project)
    assert line_edges(builder, "T", "on-board").run_time.item() == pytest.approx(600)
    assert line_edges(builder, "T2", "on-board").run_time.item() == pytest.approx(1200)


def test_negative_run_times_are_logged_and_taken_as_zero(project, tmp_path, caplog):
    """Test that a trip arriving before it left the previous stop is logged and counts as a zero run time."""
    import_feed(project, feed(("V", ["P", "Q", "S"], [20, 20], hours(7.1, 7.6))), tmp_path / "f.zip")
    with project.transit_connection as conn:
        trip_id, departure = conn.execute(
            "SELECT trip_id, departure FROM trips_schedule WHERE seq = 0 ORDER BY departure DESC LIMIT 1"
        ).fetchone()
        conn.execute(
            "UPDATE trips_schedule SET arrival = ?, departure = ? WHERE trip_id = ? AND seq = 1",
            [departure - 60, departure - 60, trip_id],
        )

    with caplog.at_level(logging.WARNING):
        builder = build(project)
    assert "arrival before departure" in caplog.text

    on_board = line_edges(builder, "V", "on-board")
    # One trip runs 20 minutes and the other "minus one minute", taken as zero
    assert on_board.run_time.iloc[0] == pytest.approx(600)
    assert (on_board.run_time >= 0).all()


def test_dwell_comes_from_the_schedule(project, tmp_path):
    """Test that dwell, boarding and alighting edges carry the scheduled dwell at their stop."""
    import_feed(project, feed(("W", ["P", "Q", "S"], [20, 20], hours(7.1, 7.6))), tmp_path / "f.zip")
    with project.transit_connection as conn:
        # Each vehicle waits 30 seconds at Q, leaving the run time from Q 30 seconds shorter
        conn.execute("UPDATE trips_schedule SET departure = arrival + 30 WHERE seq = 1")

    builder = build(project)
    stops = internal_stops(project)
    dwell = line_edges(builder, "W", "dwell")
    assert dwell.stop_id.tolist() == [stops["Q"]]
    assert dwell.dwell_time.item() == pytest.approx(30)

    boarding = line_edges(builder, "W", "boarding").set_index("stop_id").dwell_time
    alighting = line_edges(builder, "W", "alighting").set_index("stop_id").dwell_time
    assert boarding.to_dict() == {stops["P"]: 0.0, stops["Q"]: 30.0}
    assert alighting.to_dict() == {stops["Q"]: 30.0, stops["S"]: 0.0}
    assert line_edges(builder, "W", "on-board").run_time.tolist() == [1200, 1170]

    # Riders staying on pay the whole dwell, and those boarding or alighting half of it
    costs, _ = TransitCosting(alighting_penalty=0.0).edge_costs(builder.edges)
    costs = dict(zip(builder.edges.link_id, costs, strict=True))
    assert costs[dwell.link_id.item()] == pytest.approx(30)
    assert costs[line_edges(builder, "W", "boarding").link_id.iloc[1]] == pytest.approx(15)


def test_lines_without_trips_in_the_period_are_left_out(project, tmp_path):
    """Test that a line whose trips all leave outside the period has no vertices or edges in its graph."""
    import_feed(
        project, feed(("E", ["P", "Q"], [10], hours(5, 5.5)), ("M", ["Q", "S"], [10], hours(7.5))), tmp_path / "f.zip"
    )
    builder = build(project)
    assert line_edges(builder, "E", "boarding").empty
    assert not line_edges(builder, "M", "boarding").empty
