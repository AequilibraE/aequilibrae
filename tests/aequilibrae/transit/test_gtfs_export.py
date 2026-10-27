import logging
from contextlib import contextmanager
from zoneinfo import available_timezones

import pytest

from aequilibrae import Project

from .synthetic_gtfs import (
    add_stop,
    add_trip,
    add_trips,
    base_tables,
    import_feed,
    import_zip,
    read_feed,
    schedules,
)

URL = "https://example.com/syn"
TIME_ZONE = "America/Santiago"

STOPS = {"A": "Plaza", "B": "Mercado", "C": "Hospital", "D": "Terminal"}

# GTFS shape of the outbound trips of R1, through its four stops
SHAPE = [
    (-29.95, -71.33),
    (-29.9495, -71.328),
    (-29.948, -71.326),
    (-29.9475, -71.324),
    (-29.946, -71.322),
    (-29.9455, -71.32),
    (-29.944, -71.318),
]

REQUIRED_FIELDS = {
    "agency.txt": ["agency_name", "agency_url", "agency_timezone"],
    "stops.txt": ["stop_id", "stop_name", "stop_lat", "stop_lon"],
    "routes.txt": ["route_id", "route_short_name", "route_long_name", "route_type"],
    "trips.txt": ["route_id", "service_id", "trip_id"],
    "stop_times.txt": ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    "calendar_dates.txt": ["service_id", "date", "exception_type"],
}


def feed() -> dict:
    """Return a bus route with a GTFS shape, run both ways, and an extended route type running past midnight."""
    tables = base_tables()
    for i, (stop, name) in enumerate(STOPS.items()):
        add_stop(tables, stop, *SHAPE[2 * i], stop_name=name, zone_id="Z1" if i < 2 else "Z2")
    add_trips(tables, "R1", list("ABCD"), [3, 4, 3], [7 * 3600, 8 * 3600])
    add_trips(tables, "R1", list("DCBA"), [3, 4, 3], [9 * 3600])
    tables["trips.txt"][-1]["direction_id"] = 1
    # Leaves at 23:50, with a minute of dwell at C after midnight
    add_trip(tables, "N7", "night", [("A", 85800, 85800), ("C", 86460, 86520), ("D", 87000, 87000)], route_type=700)

    for trip in tables["trips.txt"][:2]:
        trip["shape_id"] = "S1"
    tables["shapes.txt"] = [
        {"shape_id": "S1", "shape_pt_lat": lat, "shape_pt_lon": lon, "shape_pt_sequence": seq}
        for seq, (lat, lon) in enumerate(SHAPE, start=1)
    ]
    return tables


def with_fares(tables: dict) -> dict:
    """Add a free fare, one with unlimited transfers and one with a transfer duration, with rules by route and zone."""

    def fare(fare_id, price, payment_method, transfers, transfer_duration=""):
        return {
            "fare_id": fare_id,
            "price": price,
            "currency_type": "CLP",
            "payment_method": payment_method,
            "transfers": transfers,
            "transfer_duration": transfer_duration,
        }

    def rule(fare_id, route_id="", origin_id="", destination_id="", contains_id=""):
        return {
            "fare_id": fare_id,
            "route_id": route_id,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "contains_id": contains_id,
        }

    tables["fare_attributes.txt"] = [fare("free", 0, 0, 0), fare("pass", 650, 1, ""), fare("one", 750.5, 0, 1, 3600)]
    tables["fare_rules.txt"] = [
        rule("one", route_id="R1"),
        rule("pass", origin_id="Z1", destination_id="Z2"),
        rule("free", route_id="N7", contains_id="Z2"),
    ]
    return tables


def line_feed(tag: str = "") -> dict:
    """Return a feed with one route over three stops in one fare zone, with IDs starting with *tag*."""
    tables = base_tables()
    for i in range(3):
        add_stop(tables, f"{tag}{i}", -29.95, -71.33 + i * 0.004, zone_id=f"{tag}Z")
    add_trips(tables, f"{tag}R", [f"{tag}{i}" for i in range(3)], [3, 4], [7 * 3600])
    return tables


def export(project, path, **kwargs) -> dict[str, list[dict]]:
    """Export the project's transit database and read the feed back."""
    kwargs = {"agency_url": URL, "agency_timezone": TIME_ZONE} | kwargs
    return read_feed(project.transit.export_gtfs(path, **kwargs))


@contextmanager
def reimported(path, folder, service_date="2026-01-05"):
    """Import an exported feed into a new project, closed on exit."""
    project = Project()
    project.new(folder)
    try:
        import_zip(project, path, service_date=service_date)
        yield project
    finally:
        project.close()


@pytest.fixture
def round_trip(empty_project, tmp_path):
    """The project with feed() imported, and a new project with its export imported."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    path = empty_project.transit.export_gtfs(tmp_path / "export.zip", agency_url=URL, agency_timezone=TIME_ZONE)
    with reimported(path, tmp_path / "copy") as copy:
        yield empty_project, copy


@contextmanager
def exporter_warnings(caplog):
    """Collect the messages the exporter logs, whether or not they propagate to the root logger."""
    logger = logging.getLogger("aequilibrae.transit.gtfs_exporter")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def fetch(project, sql: str, *params) -> list[tuple]:
    with project.transit_connection as conn:
        return conn.execute(sql, params).fetchall()


def ids(exported: dict, name: str, column: str) -> set:
    return {row[column] for row in exported.get(name, []) if row.get(column)}


def assert_foreign_keys(exported: dict) -> None:
    """Assert that every ID an exported file refers to is in the file it refers to."""
    stops = ids(exported, "stops.txt", "stop_id")
    assert ids(exported, "trips.txt", "route_id") <= ids(exported, "routes.txt", "route_id")
    assert ids(exported, "trips.txt", "service_id") <= ids(exported, "calendar_dates.txt", "service_id")
    assert ids(exported, "trips.txt", "shape_id") <= ids(exported, "shapes.txt", "shape_id")
    assert ids(exported, "stop_times.txt", "trip_id") == ids(exported, "trips.txt", "trip_id")
    assert ids(exported, "stop_times.txt", "stop_id") <= stops
    assert ids(exported, "stops.txt", "parent_station") <= stops
    assert ids(exported, "routes.txt", "agency_id") <= ids(exported, "agency.txt", "agency_id")
    assert ids(exported, "fare_attributes.txt", "agency_id") <= ids(exported, "agency.txt", "agency_id")
    assert ids(exported, "fare_rules.txt", "fare_id") <= ids(exported, "fare_attributes.txt", "fare_id")
    assert ids(exported, "fare_rules.txt", "route_id") <= ids(exported, "routes.txt", "route_id")
    for column in ["origin_id", "destination_id", "contains_id"]:
        assert ids(exported, "fare_rules.txt", column) <= ids(exported, "stops.txt", "zone_id")


def stops(project) -> dict[str, tuple]:
    sql = "SELECT stop, name, ST_X(geometry), ST_Y(geometry), transit_fare_zone FROM stops"
    return {row[0]: row[1:] for row in fetch(project, sql)}


def routes(project) -> dict[str, tuple]:
    sql = "SELECT DISTINCT route, shortname, longname, route_type FROM routes"
    return {row[0]: row[1:] for row in fetch(project, sql)}


def counts(project) -> tuple:
    sql = """SELECT (SELECT COUNT(DISTINCT route) FROM routes), (SELECT COUNT(*) FROM routes),
                    (SELECT COUNT(*) FROM trips), (SELECT COUNT(*) FROM trips_schedule)"""
    return fetch(project, sql)[0]


def test_stops_round_trip(round_trip):
    """Test that stops keep their GTFS IDs, names, coordinates and fare zones when exported and imported again."""
    original, copy = round_trip
    before, after = stops(original), stops(copy)
    assert set(after) == set(before) == set(STOPS)
    for stop, (name, lon, lat, zone) in before.items():
        assert after[stop][0] == name == STOPS[stop]
        assert after[stop][1:3] == pytest.approx((lon, lat), abs=1e-6)
        assert after[stop][3] == zone


def test_routes_round_trip(round_trip):
    """Test that routes keep their GTFS IDs, names and route types, extended ones included, and their patterns."""
    original, copy = round_trip
    assert routes(copy) == routes(original) == {"R1": ("R1", "R1", 3), "N7": ("N7", "N7", 700)}
    assert counts(copy)[:2] == counts(original)[:2] == (2, 3)


def test_trips_and_schedules_round_trip(round_trip):
    """Test that trips keep their direction and schedule, with times past midnight and a dwell, when imported again."""
    original, copy = round_trip
    directions = "SELECT trip, dir FROM trips"
    assert (
        dict(fetch(copy, directions))
        == dict(fetch(original, directions))
        == {
            "R1_0": 0,
            "R1_1": 0,
            "R1_2": 1,
            "night": 0,
        }
    )
    assert schedules(copy) == schedules(original)
    assert schedules(copy)["night"] == [(85800, 85800), (86460, 86520), (87000, 87000)]


def test_loop_route_round_trip(empty_project, tmp_path):
    """Test that a loop route, serving a pair of stops twice, keeps its stops and schedule when imported again."""
    tables = feed()
    add_trip(tables, "L1", "loop", [(stop, 7200 + 120 * i, 7200 + 120 * i + 30) for i, stop in enumerate("ABCABD")])
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    assert [row["stop_id"] for row in exported["stop_times.txt"] if row["trip_id"] == "loop"] == list("ABCABD")

    with reimported(tmp_path / "export.zip", tmp_path / "copy") as copy:
        assert schedules(copy) == schedules(empty_project)
        assert export(copy, tmp_path / "again.zip")["stop_times.txt"] == exported["stop_times.txt"]


def test_export_has_the_required_files_and_fields(empty_project, tmp_path):
    """Test that the feed has every file and field GTFS requires, filled in on every row."""
    import_feed(empty_project, with_fares(feed()), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    for name, fields in REQUIRED_FIELDS.items():
        assert exported[name], f"{name} is empty"
        for row in exported[name]:
            assert all(row[field] != "" for field in fields), f"{name} misses a required field in {row}"
    assert {"shapes.txt", "fare_attributes.txt", "fare_rules.txt"} <= set(exported)


def test_every_id_in_the_export_resolves(empty_project, tmp_path):
    """Test that every route, service, trip, stop, shape, fare and zone the export refers to is in the export."""
    import_feed(empty_project, with_fares(feed()), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    assert_foreign_keys(exported)
    assert ids(exported, "trips.txt", "shape_id") and ids(exported, "fare_rules.txt", "route_id")
    assert ids(exported, "stop_times.txt", "trip_id") == {"R1_0", "R1_1", "R1_2", "night"}


def test_agency_url_and_time_zone_are_those_given(empty_project, tmp_path):
    """Test that agency.txt has the agency's name from import and the URL and time zone the caller gives."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    assert exported["agency.txt"] == [{"agency_name": "SYN", "agency_url": URL, "agency_timezone": TIME_ZONE}]

    exported = export(empty_project, tmp_path / "export.zip", agency_url={"SYN": "http://syn.example.com"})
    assert exported["agency.txt"][0]["agency_url"] == "http://syn.example.com"


def test_agency_url_and_time_zone_are_required(empty_project, tmp_path):
    """Test that the export needs a full agency URL and a time zone, which the transit database does not store."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    transit = empty_project.transit
    with pytest.raises(TypeError):
        transit.export_gtfs(tmp_path / "export.zip")
    with pytest.raises(ValueError, match="agency_url"):
        transit.export_gtfs(tmp_path / "export.zip", agency_url="example.com", agency_timezone=TIME_ZONE)
    with pytest.raises(ValueError, match="agency_url has no entry"):
        transit.export_gtfs(tmp_path / "export.zip", agency_url={"OTHER": URL}, agency_timezone=TIME_ZONE)
    with pytest.raises(ValueError, match="agency_timezone"):
        transit.export_gtfs(tmp_path / "export.zip", agency_url=URL, agency_timezone="")
    if available_timezones():
        with pytest.raises(ValueError, match="agency_timezone"):
            transit.export_gtfs(tmp_path / "export.zip", agency_url=URL, agency_timezone="Mars/Olympus")
    assert not (tmp_path / "export.zip").exists()


def test_service_date_is_the_only_date_in_calendar_dates(empty_project, tmp_path):
    """Test that the service date imported is added in calendar_dates.txt for the agency's only service."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    assert "calendar.txt" not in exported
    assert exported["calendar_dates.txt"] == [{"service_id": "SYN_20260105", "date": "20260105", "exception_type": "1"}]
    assert ids(exported, "trips.txt", "service_id") == {"SYN_20260105"}


def test_export_fails_without_a_service_date(empty_project, tmp_path):
    """Test that the export fails if an agency with trips has no service date, as the day its trips run is unknown."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    with empty_project.transit_connection as conn:
        conn.execute("UPDATE agencies SET service_date = ''")
    with pytest.raises(ValueError, match="no valid service date"):
        export(empty_project, tmp_path / "export.zip")


def test_times_after_midnight_have_hours_past_24(empty_project, tmp_path):
    """Test that stop times are HH:MM:SS, with hours past 24 after midnight, and keep arrival and departure apart."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    night = [
        (row["arrival_time"], row["departure_time"], row["stop_id"], row["stop_sequence"])
        for row in exported["stop_times.txt"]
        if row["trip_id"] == "night"
    ]
    assert night == [
        ("23:50:00", "23:50:00", "A", "1"),
        ("24:01:00", "24:02:00", "C", "2"),
        ("24:10:00", "24:10:00", "D", "3"),
    ]
    assert {row["direction_id"] for row in exported["trips.txt"] if row["trip_id"] == "R1_2"} == {"1"}


def test_shapes_are_the_geometry_of_each_pattern(empty_project, tmp_path):
    """Test that each pattern's shape is its stored geometry: the GTFS shape imported or else its stops in sequence."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    shapes = {}
    for row in sorted(exported["shapes.txt"], key=lambda row: int(row["shape_pt_sequence"])):
        shapes.setdefault(row["shape_id"], []).extend([float(row["shape_pt_lat"]), float(row["shape_pt_lon"])])
    shape_of = {row["trip_id"]: row["shape_id"] for row in exported["trips.txt"]}

    assert len(shapes) == 3
    assert shape_of["R1_0"] == shape_of["R1_1"] != shape_of["R1_2"]
    assert shapes[shape_of["R1_0"]] == pytest.approx([x for point in SHAPE for x in point], abs=1e-9)
    stops_in_sequence = [x for i in [6, 4, 2, 0] for x in SHAPE[i]]
    assert shapes[shape_of["R1_2"]] == pytest.approx(stops_in_sequence, abs=1e-9)


def test_fares_round_trip(empty_project, tmp_path):
    """Test that fares keep price, currency, payment method, transfers, duration and rules through a round trip."""
    import_feed(empty_project, with_fares(feed()), tmp_path / "feed.zip")
    exported = export(empty_project, tmp_path / "export.zip")
    by_fare = {row["fare_id"]: row for row in exported["fare_attributes.txt"]}
    assert (by_fare["free"]["price"], by_fare["free"]["transfers"]) == ("0", "0")
    assert (by_fare["pass"]["transfers"], by_fare["pass"]["transfer_duration"]) == ("", "")
    assert (by_fare["one"]["price"], by_fare["one"]["transfer_duration"]) == ("750.5", "3600")

    attributes = "SELECT fare, price, currency, payment_method, transfer, transfer_duration FROM fare_attributes"
    rules = """SELECT a.fare, r.route, fr.origin, fr.destination, fr."contains" FROM fare_rules fr
               JOIN fare_attributes a ON a.fare_id = fr.fare_id
               LEFT JOIN (SELECT DISTINCT route_id, route FROM routes) r ON r.route_id = fr.route_id"""
    with reimported(tmp_path / "export.zip", tmp_path / "copy") as copy:
        assert {row[0]: row[1:] for row in fetch(copy, attributes)} == {
            "free": (0, "CLP", 0, 0, None),
            "pass": (650, "CLP", 1, None, None),
            "one": (750.5, "CLP", 0, 1, 3600),
        }
        assert sorted(fetch(copy, rules)) == sorted(fetch(empty_project, rules))
        assert sorted(fetch(copy, rules)) == [
            ("free", "N7", None, None, "Z2"),
            ("one", "R1", None, None, None),
            ("pass", None, "Z1", "Z2", None),
        ]


def test_parent_station_is_kept_only_if_the_parent_is_exported(empty_project, tmp_path, caplog):
    """Test that a parent_station is kept only if that parent is in the export, unserved, and others are logged."""
    tables = feed()
    add_stop(tables, "ST", -29.95, -71.331, stop_name="Station")
    for stop in tables["stops.txt"]:
        stop["parent_station"] = {"A": "ST", "B": "ST", "C": "A"}.get(stop["stop_id"], "")
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    # Import keeps only the stops trips serve, so it leaves the station out
    with exporter_warnings(caplog):
        exported = export(empty_project, tmp_path / "export.zip")
    assert "parent_station" not in exported["stops.txt"][0] and "location_type" not in exported["stops.txt"][0]
    assert any(message.startswith("3 stops are exported without their parent_station") for message in caplog.messages)

    (agency_id,) = fetch(empty_project, "SELECT agency_id FROM agencies")[0]
    with empty_project.transit_connection as conn:
        sql = """INSERT INTO stops (stop_id, stop, agency_id, name, route_type, geometry)
                 VALUES ('999', 'ST', ?, 'Station', -1, MakePoint(-71.331, -29.95, 4326))"""
        conn.execute(sql, [agency_id])
    caplog.clear()
    with exporter_warnings(caplog):
        exported = export(empty_project, tmp_path / "export.zip")
    by_stop = {row["stop_id"]: (row["location_type"], row["parent_station"]) for row in exported["stops.txt"]}
    assert by_stop == {"A": ("0", "ST"), "B": ("0", "ST"), "C": ("0", ""), "D": ("0", ""), "ST": ("1", "")}
    # The parent of C is a stop trips serve, which cannot be a station
    assert any(message.startswith("1 stops are exported without their parent_station") for message in caplog.messages)
    assert_foreign_keys(exported)


def test_several_agencies_are_exported_with_agency_ids(empty_project, tmp_path):
    """Test that agencies exported together have agency_id columns and a service each, or one agency on its own."""
    import_feed(empty_project, line_feed("a"), tmp_path / "a.zip", agency="A")
    import_feed(empty_project, line_feed("b"), tmp_path / "b.zip", agency="B")
    urls = {"A": "https://a.example.com", "B": "https://b.example.com"}

    exported = export(empty_project, tmp_path / "export.zip", agency_url=urls)
    assert exported["agency.txt"] == [
        {"agency_id": name, "agency_name": name, "agency_url": url, "agency_timezone": TIME_ZONE}
        for name, url in urls.items()
    ]
    assert {row["route_id"]: row["agency_id"] for row in exported["routes.txt"]} == {"aR": "A", "bR": "B"}
    assert {row["trip_id"]: row["service_id"] for row in exported["trips.txt"]} == {
        "aR_0": "A_20260105",
        "bR_0": "B_20260105",
    }
    assert ids(exported, "calendar_dates.txt", "service_id") == {"A_20260105", "B_20260105"}
    assert_foreign_keys(exported)

    (b_id,) = fetch(empty_project, "SELECT agency_id FROM agencies WHERE agency = 'B'")[0]
    for agencies in (["B"], [b_id]):
        exported = export(empty_project, tmp_path / "export.zip", agency_url=urls, agencies=agencies)
        assert exported["agency.txt"] == [{"agency_name": "B", "agency_url": urls["B"], "agency_timezone": TIME_ZONE}]
        assert ids(exported, "stops.txt", "stop_id") == {"b0", "b1", "b2"}
        assert "agency_id" not in exported["routes.txt"][0]

    with pytest.raises(ValueError, match="not in the transit database"):
        export(empty_project, tmp_path / "export.zip", agencies=["C"])


def test_agencies_sharing_ids_need_prefixes(empty_project, tmp_path):
    """Test that agencies sharing GTFS IDs are exported together only with IDs prefixed by their agency."""
    import_feed(empty_project, line_feed(), tmp_path / "a.zip", agency="A")
    import_feed(empty_project, line_feed(), tmp_path / "b.zip", agency="B")
    with pytest.raises(ValueError, match="share GTFS IDs.*stop_id '0', '1', '2'.*prefix_ids=True"):
        export(empty_project, tmp_path / "export.zip")
    assert not (tmp_path / "export.zip").exists()

    exported = export(empty_project, tmp_path / "export.zip", prefix_ids=True)
    assert ids(exported, "stops.txt", "stop_id") == {f"{agency}:{i}" for agency in "AB" for i in range(3)}
    assert ids(exported, "stops.txt", "zone_id") == {"A:Z", "B:Z"}
    assert {row["route_id"]: row["agency_id"] for row in exported["routes.txt"]} == {"A:R": "A", "B:R": "B"}
    assert ids(exported, "stop_times.txt", "trip_id") == {"A:R_0", "B:R_0"}
    assert_foreign_keys(exported)


def test_trips_of_patterns_with_broken_route_links_are_left_out(empty_project, tmp_path, caplog):
    """Test that trips of a pattern whose route links do not join up are left out, with a warning."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    (pattern_id,) = fetch(empty_project, "SELECT DISTINCT t.pattern_id FROM trips t WHERE trip = 'R1_0'")[0]
    with empty_project.transit_connection as conn:
        conn.execute("DELETE FROM route_links WHERE pattern_id = ? AND seq = 1", [pattern_id])

    with exporter_warnings(caplog):
        exported = export(empty_project, tmp_path / "export.zip")
    assert ids(exported, "trips.txt", "trip_id") == {"R1_2", "night"}
    assert any(f"patterns '{pattern_id}' do not join up" in message for message in caplog.messages)
    assert_foreign_keys(exported)


def test_coquimbo_exports_and_imports_again(coquimbo_example, tmp_path):
    """Test that Coquimbo exports and imports again with the same routes, patterns, trips and stop events."""
    path = tmp_path / "coquimbo_export.zip"
    exported = export(coquimbo_example, path)
    assert_foreign_keys(exported)
    assert len(ids(exported, "shapes.txt", "shape_id")) == counts(coquimbo_example)[1]

    (service_date,) = fetch(coquimbo_example, "SELECT service_date FROM agencies")[0]
    with reimported(path, tmp_path / "copy", service_date=service_date) as copy:
        assert counts(copy) == counts(coquimbo_example)
        assert schedules(copy) == schedules(coquimbo_example)
