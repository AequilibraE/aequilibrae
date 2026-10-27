import logging

import numpy as np
import pandas as pd
import pytest

from aequilibrae.utils.geo_utils import haversine

from .synthetic_gtfs import add_stop, add_trip, base_tables, hms, import_feed, internal_stops, schedules

LAT, LON, STEP = -29.95, -71.33, 0.004


def sec(clock: str) -> int:
    """Return the seconds after midnight of a clock time such as "7:03" or "25:05:30"."""
    hours, minutes, seconds = (clock.split(":") + ["0"])[:3]
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def events(text: str) -> list[tuple[str, int, int]]:
    """Return ``(stop, arrival, departure)`` events from text such as "A 7:00, B 7:03 7:04", departing on arrival."""
    stops = [part.split() for part in text.split(",")]
    return [(stop, sec(arrival), sec(departure[0] if departure else arrival)) for stop, arrival, *departure in stops]


def times(trip_events: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    return [(arrival, departure) for _, arrival, departure in trip_events]


def line_feed(stops="ABCDE") -> dict:
    """Return a feed with its stops a little under 400 m apart, from west to east."""
    tables = base_tables()
    for i, stop in enumerate(stops):
        add_stop(tables, stop, LAT, LON + i * STEP)
    return tables


def imported(project, tmp_path, *trips, **builder_calls) -> dict:
    """Import a trip of route R for each list of events and return their schedules."""
    tables = line_feed()
    for trip_id, trip_events in trips:
        add_trip(tables, "R", trip_id, trip_events)
    import_feed(project, tables, tmp_path / "feed.zip", **builder_calls)
    return schedules(project)


def route_links(project) -> dict[str, list[tuple[str, str]]]:
    """Return the GTFS stops of the route links of each imported trip's pattern, keyed by GTFS trip ID."""
    gtfs_stop = {internal: stop for stop, internal in internal_stops(project).items()}
    sql = "SELECT t.trip, l.from_stop, l.to_stop FROM trips t JOIN route_links l USING (pattern_id) ORDER BY l.seq"
    out: dict[str, list[tuple[str, str]]] = {}
    with project.transit_connection as conn:
        for trip, from_stop, to_stop in conn.execute(sql).fetchall():
            out.setdefault(trip, []).append((gtfs_stop[str(from_stop)], gtfs_stop[str(to_stop)]))
    return out


@pytest.mark.parametrize(
    "gtfs, stored",
    [
        ("A 7:00, B 7:03 7:04:30, D 7:12", "A 7:00, B 7:03 7:04:30, D 7:12"),
        ("A 7:00, B 7:03, C 7:03, D 7:02", "A 7:00, B 7:03, C 7:03:01, D 7:03:02"),
        ("A 7:00, B 7:03, B 7:05 7:06, C 7:09", "A 7:00, B 7:03 7:06, C 7:09"),
        ("A 24:50, B 25:05 25:06", "A 24:50, B 25:05 25:06"),
        ("A 23:58, B 23:59:30 0:00:30, C 0:02", "A 23:58, B 23:59:30 24:00:30, C 24:02"),
    ],
)
def test_stop_times_are_imported_with_their_dwell_and_positive_run_times(empty_project, tmp_path, gtfs, stored):
    """Test that stop times keep their scheduled dwell, with positive run times and times going past midnight."""
    assert imported(empty_project, tmp_path, ("T", events(gtfs))) == {"T": times(events(stored))}


def test_missing_times_are_filled(empty_project, tmp_path):
    """Test that a stop with one time uses it for both, and one without times follows the previous departure."""
    tables = line_feed()
    add_trip(tables, "R", "T", events("A 7:00, B 7:03 7:04, C 0:00, D 7:09"))
    rows = tables["stop_times.txt"]
    rows[1]["arrival_time"] = rows[2]["arrival_time"] = rows[2]["departure_time"] = ""
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    expected = events("A 7:00, B 7:04, C 7:04:01, D 7:09:01")
    assert schedules(empty_project) == {"T": times(expected)}


def test_untimed_trip_ends_and_unknown_stops_are_dropped(empty_project, tmp_path, caplog):
    """Test that trips without a time at an end are skipped and stop times at unknown stops dropped, with warnings."""
    tables = line_feed()
    trips = {"first": "A 7:00, B 7:03", "last": "A 8:00, B 8:03", "unknown": "A 9:00, X 9:02, C 9:05"}
    for trip_id, text in trips.items():
        add_trip(tables, "R", trip_id, events(text))
    first, last = tables["stop_times.txt"][0], tables["stop_times.txt"][3]
    first["arrival_time"] = first["departure_time"] = last["arrival_time"] = last["departure_time"] = ""
    with caplog.at_level(logging.WARNING, logger="aequilibrae"):
        import_feed(empty_project, tables, tmp_path / "feed.zip")
    assert schedules(empty_project) == {"unknown": times(events("A 9:00, C 9:05"))}
    assert "first or last stop time were skipped: first, last" in caplog.text
    assert "absent from stops.txt were dropped: X" in caplog.text


def test_repaired_times_are_logged(empty_project, tmp_path, caplog):
    """Test that the number of departures before their arrival and of non-positive run times repaired is logged."""
    with caplog.at_level(logging.WARNING, logger="aequilibrae"):
        imported(empty_project, tmp_path, ("T", events("A 7:00, B 7:05 7:04, C 7:03, D 7:03")))
    assert "1 departures before their arrival were repaired" in caplog.text
    assert "2 run times of zero or less were repaired" in caplog.text


def test_maximum_speeds_are_applied(empty_project, tmp_path):
    """Test that a run faster than the maximum speed for its distance is slowed down to it, keeping the dwell."""
    trip = events("A 7:00, B 7:02 7:02:30, C 7:03")
    max_speeds = pd.DataFrame({"mode": [3], "min_distance": [0], "max_distance": [1000], "speed": [8.0]})
    run = int(np.ceil(haversine(LON + STEP, LAT, LON + 2 * STEP, LAT) / 8))
    schedule = imported(empty_project, tmp_path, ("T", trip), set_maximum_speeds=max_speeds)
    assert schedule == {"T": times(trip[:2]) + [(sec("7:02:30") + run,) * 2]}


def test_frequency_windows_repeat_their_trip(empty_project, tmp_path):
    """Test that each frequencies.txt window imports one trip per headway until its end, and not the template."""
    tables = line_feed()
    add_trip(tables, "R", "F", [("A", 0, 0), ("B", 180, 240), ("C", 480, 480)])
    for start, end, headway in [(sec("16:00"), sec("17:00"), 1800), (sec("7:00"), sec("7:30"), 600)]:
        row = {"trip_id": "F", "start_time": hms(start), "end_time": hms(end), "headway_secs": headway}
        tables.setdefault("frequencies.txt", []).append(row)
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    starts = [sec(clock) for clock in ["7:00", "7:10", "7:20", "16:00", "16:30"]]
    assert schedules(empty_project) == {f"F-{s}": [(s, s), (s + 180, s + 240), (s + 480, s + 480)] for s in starts}


def test_frequency_trip_ids_do_not_replace_trips_of_the_feed(empty_project, tmp_path):
    """Test that a trip of a frequency window gets another ID when the feed already has a trip with its ID."""
    tables = line_feed()
    add_trip(tables, "R", "F", events("A 0:00, B 0:03"))
    add_trip(tables, "R", "F-25200", events("C 7:00, D 7:05"))
    tables["frequencies.txt"] = [{"trip_id": "F", "start_time": "7:00:00", "end_time": "7:05:00", "headway_secs": 600}]
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    assert schedules(empty_project) == {
        "F-25200": times(events("C 7:00, D 7:05")),
        "F-25200-copy": times(events("A 7:00, B 7:03")),
    }


def test_loop_trip_keeps_its_repeated_stop_pairs(empty_project, tmp_path):
    """Test that a trip through the same pair of stops twice keeps both in trips_schedule and route_links."""
    trip = [(stop, sec("7:00") + 120 * i, sec("7:00") + 120 * i + 30) for i, stop in enumerate("ABCABD")]
    assert imported(empty_project, tmp_path, ("loop", trip)) == {"loop": times(trip)}
    assert route_links(empty_project) == {"loop": list(zip("ABCAB", "BCABD", strict=True))}


def test_links_to_a_stop_visited_again_follow_the_shape(empty_project, tmp_path):
    """Test that the link back to the first stop of a loop follows the bend of its shape, not a straight line."""
    tables = base_tables()
    corners = {"A": (LAT, LON), "B": (LAT, LON + 0.01), "C": (LAT - 0.01, LON + 0.01), "D": (LAT - 0.01, LON)}
    for stop, (lat, lon) in corners.items():
        add_stop(tables, stop, lat, lon)
    points = [*corners.values(), (LAT - 0.005, LON - 0.005), corners["A"]]
    columns = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    tables["shapes.txt"] = [dict(zip(columns, ["S", *point, i], strict=True)) for i, point in enumerate(points)]
    add_trip(tables, "L", "loop", events("A 7:00, B 7:02, C 7:04, D 7:06, A 7:08"))
    tables["trips.txt"][0]["shape_id"] = "S"
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    with empty_project.transit_connection as conn:
        points = conn.execute("SELECT ST_NumPoints(geometry) FROM route_links ORDER BY seq").fetchall()
    assert points == [(2,), (2,), (2,), (3,)]


def test_stop_ids_that_concatenate_alike_make_different_patterns(empty_project, tmp_path):
    """Test that trips through stops "1", "23" and "12", "3" of one route get a stop pattern each."""
    tables = base_tables()
    for i, stop in enumerate(["1", "23", "12", "3", "4"]):
        add_stop(tables, stop, LAT, LON + i * STEP)
    add_trip(tables, "R", "X", events("1 7:00, 23 7:02, 4 7:05"))
    add_trip(tables, "R", "Y", events("12 8:00, 3 8:02, 4 8:05"))
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    assert route_links(empty_project) == {"X": [("1", "23"), ("23", "4")], "Y": [("12", "3"), ("3", "4")]}
