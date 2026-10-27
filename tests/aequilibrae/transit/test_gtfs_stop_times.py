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


def test_loop_trip_keeps_its_repeated_stop_pairs(empty_project, tmp_path):
    """Test that a trip through the same pair of stops twice keeps both in trips_schedule and route_links."""
    trip = [(stop, sec("7:00") + 120 * i, sec("7:00") + 120 * i + 30) for i, stop in enumerate("ABCABD")]
    assert imported(empty_project, tmp_path, ("loop", trip)) == {"loop": times(trip)}
    assert route_links(empty_project) == {"loop": list(zip("ABCAB", "BCABD", strict=True))}


def test_stop_ids_that_concatenate_alike_make_different_patterns(empty_project, tmp_path):
    """Test that trips through stops "1", "23" and "12", "3" of one route get a stop pattern each."""
    tables = base_tables()
    for i, stop in enumerate(["1", "23", "12", "3", "4"]):
        add_stop(tables, stop, LAT, LON + i * STEP)
    add_trip(tables, "R", "X", events("1 7:00, 23 7:02, 4 7:05"))
    add_trip(tables, "R", "Y", events("12 8:00, 3 8:02, 4 8:05"))
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    assert route_links(empty_project) == {"X": [("1", "23"), ("23", "4")], "Y": [("12", "3"), ("3", "4")]}
