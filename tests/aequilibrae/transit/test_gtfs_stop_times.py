import logging

import numpy as np
import pandas as pd
import pytest

from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.geo_utils import haversine

from .synthetic_gtfs import (
    add_frequency,
    add_stop,
    add_trip,
    base_tables,
    hms,
    import_feed,
    internal_stops,
    schedules,
)

LAT, LON, STEP = -29.95, -71.33, 0.004


def sec(clock: str) -> int:
    """Return the seconds after midnight of a clock time such as "7:03" or "25:05:30"."""
    hours, minutes, seconds = (clock.split(":") + ["0"])[:3]
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def events(*stops: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Return ``(stop, arrival, departure)`` events from clock times, departing on arrival unless told otherwise."""
    return [(stop, sec(arrival), sec(departure[0] if departure else arrival)) for stop, arrival, *departure in stops]


def times(trip_events: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    """Return the ``(arrival, departure)`` pairs of a trip's events."""
    return [(arrival, departure) for _, arrival, departure in trip_events]


def dwelling_at_b(start: int) -> list[tuple[str, int, int]]:
    """Return the events of a trip leaving A at ``start``, in seconds, dwelling a minute at B on its way to C."""
    return [("A", start, start), ("B", start + 180, start + 240), ("C", start + 480, start + 480)]


def line_feed(stops="ABCDE") -> dict:
    """Return a feed with its stops a little under 400 m apart, from west to east."""
    tables = base_tables()
    for i, stop in enumerate(stops):
        add_stop(tables, stop, LAT, LON + i * STEP)
    return tables


def source_times(gtfs) -> dict[str, list[int]]:
    """Return the source of the times at each stop of the trips imported, keyed by GTFS trip ID."""
    return {trip.trip: [int(x) for x in trip.source_time] for trip in gtfs.select_trips}


def route_links(project) -> dict[str, list[tuple[str, str]]]:
    """Return the GTFS stops of the route links of each imported trip's pattern, keyed by GTFS trip ID."""
    gtfs_stop = {internal: stop for stop, internal in internal_stops(project).items()}
    sql = """SELECT t.trip, l.from_stop, l.to_stop FROM trips t JOIN route_links l ON t.pattern_id = l.pattern_id
             ORDER BY t.trip, l.seq"""
    out: dict[str, list[tuple[str, str]]] = {}
    with project.transit_connection as conn:
        for trip, from_stop, to_stop in conn.execute(sql).fetchall():
            out.setdefault(trip, []).append((gtfs_stop[str(from_stop)], gtfs_stop[str(to_stop)]))
    return out


def test_scheduled_dwell_is_kept(empty_project, tmp_path):
    """Test that the scheduled dwell at each stop is kept in trips_schedule."""
    tables = line_feed()
    trip = events(("A", "7:00"), ("B", "7:03", "7:04:30"), ("C", "7:08", "7:09"), ("D", "7:12"))
    add_trip(tables, "R", "T", trip)
    gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert schedules(empty_project) == {"T": times(trip)}
    assert source_times(gtfs) == {"T": [0, 0, 0, 0]}


def test_missing_times_are_filled(empty_project, tmp_path):
    """Test that a stop with one time uses it for both, and one without times follows the previous departure."""
    tables = line_feed()
    add_trip(tables, "R", "T", events(("A", "7:00"), ("B", "7:03", "7:04"), ("C", "0:00"), ("D", "7:09")))
    rows = tables["stop_times.txt"]
    rows[1]["arrival_time"] = ""
    rows[2]["arrival_time"] = rows[2]["departure_time"] = ""
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    # C takes the departure from B and is then de-conflicted, shifting the rest of the trip by one second
    expected = events(("A", "7:00"), ("B", "7:04"), ("C", "7:04:01"), ("D", "7:09:01"))
    assert schedules(empty_project) == {"T": times(expected)}


def test_consecutive_equal_times_are_deconflicted(empty_project, tmp_path):
    """Test that stops reached at the time the previous one is left get one second of run time each."""
    tables = line_feed()
    trip = events(("A", "7:00"), ("B", "7:02", "7:03"), ("C", "7:03"), ("D", "7:03"), ("E", "7:06"))
    add_trip(tables, "R", "T", trip)
    gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    expected = events(("A", "7:00"), ("B", "7:02", "7:03"), ("C", "7:03:01"), ("D", "7:03:02"), ("E", "7:06:02"))
    assert schedules(empty_project) == {"T": times(expected)}
    assert source_times(gtfs) == {"T": [0, 0, 1, 1, 1]}


def test_maximum_speeds_are_applied(empty_project, tmp_path):
    """Test that a run faster than the maximum speed for its distance is slowed down to it, keeping the dwell."""
    tables = line_feed()
    trip = events(("A", "7:00"), ("B", "7:02", "7:02:30"), ("C", "7:03"), ("D", "7:05"))
    add_trip(tables, "R", "T", trip)
    max_speeds = pd.DataFrame({"mode": [3], "min_distance": [0], "max_distance": [1000], "speed": [8.0]})
    gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip", set_maximum_speeds=max_speeds)

    # Only B to C, a little under 400 m in 30 s, is faster than 8 m/s. D is still two minutes after C
    run = int(np.ceil(haversine(LON + STEP, LAT, LON + 2 * STEP, LAT) / 8))
    arrive_c = sec("7:02:30") + run
    assert run > 30
    assert schedules(empty_project) == {"T": times(trip[:2]) + [(arrive_c, arrive_c), (arrive_c + 120, arrive_c + 120)]}
    assert source_times(gtfs) == {"T": [0, 0, 2, 2]}


def test_negative_run_time_is_repaired_and_logged(empty_project, tmp_path, caplog):
    """Test that arriving at a stop before leaving the previous one is repaired like a tie, with a warning."""
    tables = line_feed()
    add_trip(tables, "R", "backwards", events(("A", "7:00"), ("B", "7:05", "7:06"), ("C", "7:04"), ("D", "7:10")))
    with caplog.at_level(logging.WARNING, logger="aequilibrae.transit.gtfs_loader"):
        gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    expected = events(("A", "7:00"), ("B", "7:05", "7:06"), ("C", "7:06:01"), ("D", "7:12:01"))
    assert schedules(empty_project) == {"backwards": times(expected)}
    assert source_times(gtfs) == {"backwards": [0, 0, 1, 1]}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "backwards" in r.getMessage()]
    assert len(warnings) == 1


def test_departure_before_arrival_is_repaired_and_logged(empty_project, tmp_path, caplog):
    """Test that departing a stop before arriving at it becomes a dwell of zero, with a warning."""
    tables = line_feed()
    add_trip(tables, "R", "early", events(("A", "7:00"), ("B", "7:05", "7:04"), ("C", "7:08")))
    with caplog.at_level(logging.WARNING, logger="aequilibrae.transit.gtfs_loader"):
        gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert schedules(empty_project) == {"early": times(events(("A", "7:00"), ("B", "7:05"), ("C", "7:08")))}
    assert source_times(gtfs) == {"early": [0, 1, 0]}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "early" in r.getMessage()]
    assert len(warnings) == 1


def test_frequency_window_repeats_its_trip(empty_project, tmp_path):
    """Test that a frequencies.txt window imports one trip per headway, until its end, and not the template trip."""
    tables = line_feed()
    add_trip(tables, "R", "F", dwelling_at_b(sec("6:00")))
    add_frequency(tables, "F", sec("7:00"), sec("8:00"), 900)
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    starts = [sec(clock) for clock in ["7:00", "7:15", "7:30", "7:45"]]
    expected = {f"F-{hms(start).replace(':', '')}": times(dwelling_at_b(start)) for start in starts}
    assert sorted(expected) == ["F-070000", "F-071500", "F-073000", "F-074500"]
    assert schedules(empty_project) == expected
    with empty_project.transit_connection as conn:
        assert conn.execute("SELECT COUNT(DISTINCT trip_id), COUNT(*) FROM trips").fetchone() == (4, 4)


def test_two_frequency_windows_for_one_trip(empty_project, tmp_path):
    """Test that every frequencies.txt window of a trip is imported, next to trips without frequencies."""
    tables = line_feed()
    add_trip(tables, "R", "F", dwelling_at_b(sec("6:00")))
    add_trip(tables, "R", "S", dwelling_at_b(sec("12:00")))
    add_frequency(tables, "F", sec("16:00"), sec("17:00"), 1800)
    add_frequency(tables, "F", sec("7:00"), sec("7:30"), 600)
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    starts = [sec(clock) for clock in ["7:00", "7:10", "7:20", "16:00", "16:30"]]
    expected = {f"F-{hms(start).replace(':', '')}": times(dwelling_at_b(start)) for start in starts}
    assert schedules(empty_project) == expected | {"S": times(dwelling_at_b(sec("12:00")))}


@pytest.mark.parametrize("stops", ["ABCBCD", "ABCABC"])
def test_loop_trip_keeps_its_repeated_stop_pairs(empty_project, tmp_path, stops):
    """Test that a trip through the same pair of stops twice keeps both in trips_schedule and route_links."""
    tables = line_feed()
    trip = [(stop, sec("7:00") + 120 * i, sec("7:00") + 120 * i + 30) for i, stop in enumerate(stops)]
    add_trip(tables, "R", "loop", trip)
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert schedules(empty_project) == {"loop": times(trip)}
    assert route_links(empty_project) == {"loop": list(zip(stops[:-1], stops[1:], strict=True))}


def test_consecutive_records_at_a_stop_become_one_dwell(empty_project, tmp_path):
    """Test that consecutive stop_times at one stop merge into a stop from its first arrival to its last departure."""
    tables = line_feed()
    add_trip(tables, "R", "T", events(("A", "7:00"), ("B", "7:03"), ("B", "7:05", "7:06"), ("C", "7:09")))
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert schedules(empty_project) == {"T": times(events(("A", "7:00"), ("B", "7:03", "7:06"), ("C", "7:09")))}
    assert route_links(empty_project) == {"T": [("A", "B"), ("B", "C")]}


def test_stop_ids_that_concatenate_alike_make_different_patterns(empty_project, tmp_path):
    """Test that trips through stops "1", "23" and "12", "3" of one route get a stop pattern each."""
    tables = base_tables()
    for i, stop in enumerate(["1", "23", "12", "3", "4"]):
        add_stop(tables, stop, LAT, LON + i * STEP)
    add_trip(tables, "R", "X", events(("1", "7:00"), ("23", "7:02"), ("4", "7:05")))
    add_trip(tables, "R", "Y", events(("12", "8:00"), ("3", "8:02"), ("4", "8:05")))
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    with empty_project.transit_connection as conn:
        patterns = dict(conn.execute("SELECT trip, pattern_id FROM trips").fetchall())
    assert patterns["X"] != patterns["Y"]
    assert route_links(empty_project) == {"X": [("1", "23"), ("23", "4")], "Y": [("12", "3"), ("3", "4")]}


def test_trip_after_midnight_is_stored_unchanged(empty_project, tmp_path):
    """Test that a trip with times past 24:00:00 keeps them in trips_schedule."""
    tables = line_feed()
    trip = events(("A", "24:50"), ("B", "25:05", "25:06"), ("C", "25:20"))
    add_trip(tables, "R", "late", trip)
    gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert schedules(empty_project) == {"late": times(trip)}
    assert source_times(gtfs) == {"late": [0, 0, 0]}


@pytest.mark.parametrize(
    "gtfs_b, stored_b, sources",
    [
        (("23:58", "23:59:30"), ("23:58", "23:59:30"), [0, 0, 1, 1]),
        (("23:59:30", "0:00:30"), ("23:59:30", "24:00:30"), [0, 1, 1, 1]),
    ],
)
def test_clock_times_crossing_midnight_continue_past_24(empty_project, tmp_path, caplog, gtfs_b, stored_b, sources):
    """Test that times going back to 00:xx after 23:xx are read as past 24:00:00, with a warning."""
    tables = line_feed()
    add_trip(tables, "R", "night", events(("A", "23:50"), ("B", *gtfs_b), ("C", "0:02"), ("D", "0:10")))
    with caplog.at_level(logging.WARNING, logger="aequilibrae.transit.gtfs_loader"):
        gtfs = import_feed(empty_project, tables, tmp_path / "feed.zip")

    expected = events(("A", "23:50"), ("B", *stored_b), ("C", "24:02"), ("D", "24:10"))
    assert schedules(empty_project) == {"night": times(expected)}
    assert source_times(gtfs) == {"night": sources}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "night" in r.getMessage()]
    assert len(warnings) == 1


def test_graph_run_times_exclude_scheduled_dwell(empty_project, tmp_path):
    """Test that transit graph segment times run from leaving a stop to reaching the next, without its dwell."""
    tables = line_feed()
    add_trip(tables, "R", "T", dwelling_at_b(sec("7:00")))
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    travel_times = TransitGraphBuilder(empty_project)._compute_segment_travel_time(time_filter=False)
    assert travel_times.sort_values("seq").trav_time.tolist() == [180, 240]
