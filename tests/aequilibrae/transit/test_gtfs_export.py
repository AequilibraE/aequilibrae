import pytest

from aequilibrae import Project

from .synthetic_gtfs import add_stop, add_trip, add_trips, base_tables, import_feed, import_zip, read_feed, schedules

URL, TIME_ZONE = "https://example.com/syn", "America/Santiago"
STOPS = {"A": "Plaza", "B": "Mercado", "C": "Hospital", "D": "Terminal"}


def feed() -> dict:
    """Return a route run both ways, a loop, a night route of extended type with dwell past midnight and fares."""
    tables = base_tables()
    for i, (stop, name) in enumerate(STOPS.items()):
        add_stop(tables, stop, -29.95, -71.33 + i * 0.004, stop_name=name, zone_id="Z1" if i < 2 else "Z2")
    add_trips(tables, "R1", list("ABCD"), [3, 4, 3], [7 * 3600, 8 * 3600])
    add_trips(tables, "R1", list("DCBA"), [3, 4, 3], [9 * 3600])
    tables["trips.txt"][-1]["direction_id"] = 1
    add_trip(tables, "N7", "night", [("A", 85800, 85800), ("C", 86460, 86520), ("D", 87000, 87000)], route_type=700)
    add_trip(tables, "L1", "loop", [(stop, 7200 + 120 * i, 7230 + 120 * i) for i, stop in enumerate("ABCABD")])

    fare = ["fare_id", "price", "currency_type", "payment_method", "transfers", "transfer_duration"]
    rule = ["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]
    fares = [["free", 0, "CLP", 0, 0, ""], ["pass", 650, "CLP", 1, "", ""], ["one", 750.5, "CLP", 0, 1, 3600]]
    rules = [["one", "R1", "", "", ""], ["pass", "", "Z1", "Z2", ""], ["free", "N7", "", "", "Z2"]]
    tables["fare_attributes.txt"] = [dict(zip(fare, row, strict=True)) for row in fares]
    tables["fare_rules.txt"] = [dict(zip(rule, row, strict=True)) for row in rules]
    return tables


def fetch(project, sql: str) -> list[tuple]:
    with project.transit_connection as conn:
        return sorted(conn.execute(sql).fetchall(), key=str)


def export_and_import(project, tmp_path) -> Project:
    """Export the transit services of the project and return a new project with the export imported."""
    project.transit.export_gtfs(tmp_path / "export.zip", URL, TIME_ZONE)
    copy = Project()
    copy.new(tmp_path / "copy")
    import_zip(copy, tmp_path / "export.zip")
    return copy


@pytest.fixture
def round_trip(empty_project, tmp_path):
    """The project with feed() imported, its export, and a new project with the export imported."""
    import_feed(empty_project, feed(), tmp_path / "feed.zip")
    copy = export_and_import(empty_project, tmp_path)
    yield empty_project, read_feed(tmp_path / "export.zip"), copy
    copy.close()


def test_stops_round_trip(round_trip):
    """Test that stops keep their GTFS IDs, names, coordinates and fare zones when exported and imported again."""
    original, _, copy = round_trip
    sql = "SELECT stop, name, ROUND(ST_X(geometry), 6), ROUND(ST_Y(geometry), 6), transit_fare_zone FROM stops"
    assert fetch(copy, sql) == fetch(original, sql)
    assert {row[0]: row[1] for row in fetch(copy, sql)} == STOPS


def test_routes_round_trip(round_trip):
    """Test that routes keep their GTFS IDs, names and route types, extended ones included."""
    original, _, copy = round_trip
    sql = "SELECT DISTINCT route, shortname, longname, route_type FROM routes"
    assert fetch(copy, sql) == fetch(original, sql)
    assert {row[0]: row[3] for row in fetch(copy, sql)} == {"R1": 3, "N7": 700, "L1": 3}


def test_trips_and_schedules_round_trip(round_trip):
    """Test that trips keep their direction and schedule, including a loop and a dwell past midnight."""
    original, _, copy = round_trip
    assert fetch(copy, "SELECT trip, dir FROM trips") == fetch(original, "SELECT trip, dir FROM trips")
    assert schedules(copy) == schedules(original)
    assert schedules(copy)["night"] == [(85800, 85800), (86460, 86520), (87000, 87000)]


def test_fares_round_trip(round_trip):
    """Test that fares and their rules by route and by zone are the same when exported and imported again."""
    original, _, copy = round_trip
    fares = "SELECT fare, price, currency, payment_method, transfer, transfer_duration FROM fare_attributes"
    rules = """SELECT a.fare, r.route, fr.origin, fr.destination, fr."contains" FROM fare_rules fr
               JOIN fare_attributes a USING (fare_id) LEFT JOIN (SELECT DISTINCT route_id, route FROM routes) r
               USING (route_id)"""
    assert fetch(copy, fares) == fetch(original, fares)
    assert fetch(copy, rules) == fetch(original, rules)


def test_accented_names_round_trip(empty_project, tmp_path):
    """Test that IDs, names and fare zones with accents and other non-ASCII characters survive import and export."""
    tables = base_tables()
    names = {"Ñ1": "Peñuelas", "Ç2": "Estación São João — 1"}
    for i, (stop, name) in enumerate(names.items()):
        add_stop(tables, stop, -29.95, -71.33 + i * 0.004, stop_name=name, zone_id="Zona Ñ")
    add_trips(tables, "Línea 1", list(names), [3], [7 * 3600])
    tables["routes.txt"][0]["route_long_name"] = "Coquimbo – La Serena"
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    copy = export_and_import(empty_project, tmp_path)
    for project in [empty_project, copy]:
        stops = fetch(project, "SELECT stop, name, transit_fare_zone FROM stops")
        assert stops == [("Ç2", "Estación São João — 1", "Zona Ñ"), ("Ñ1", "Peñuelas", "Zona Ñ")]
        assert fetch(project, "SELECT DISTINCT route, longname FROM routes") == [("Línea 1", "Coquimbo – La Serena")]
    copy.close()


def test_export_has_agency_service_and_times_past_midnight(round_trip):
    """Test that the export has the agency URL and time zone given, its service date and times past 24:00:00."""
    _, exported, _ = round_trip
    assert [(a["agency_url"], a["agency_timezone"]) for a in exported["agency.txt"]] == [(URL, TIME_ZONE)]
    assert [row["date"] for row in exported["calendar_dates.txt"]] == ["20260105"]
    night = [row["departure_time"] for row in exported["stop_times.txt"] if row["trip_id"] == "night"]
    assert night == ["23:50:00", "24:02:00", "24:10:00"]


def test_coquimbo_exports_and_imports_again(coquimbo_example, tmp_path):
    """Test that the Coquimbo example imports again from its export with the same routes, trips and schedules."""
    coquimbo_example.transit.export_gtfs(tmp_path / "export.zip", URL, TIME_ZONE)
    counts = "SELECT (SELECT COUNT(*) FROM routes), (SELECT COUNT(*) FROM trips), (SELECT COUNT(*) FROM trips_schedule)"
    (service_date,) = fetch(coquimbo_example, "SELECT service_date FROM agencies")[0]
    before = fetch(coquimbo_example, counts), schedules(coquimbo_example)

    copy = Project()
    copy.new(tmp_path / "copy")
    import_zip(copy, tmp_path / "export.zip", service_date=service_date)
    assert (fetch(copy, counts), schedules(copy)) == before
    copy.close()


def test_agencies_sharing_ids_are_exported_with_prefixes(empty_project, tmp_path):
    """Test that agencies sharing GTFS IDs are exported together only with IDs prefixed by their agency."""
    agencies = ["NORTE", "SUR"]
    for agency in agencies:
        import_feed(empty_project, feed(), tmp_path / f"{agency}.zip", agency=agency)
    with pytest.raises(ValueError, match="prefix_ids=True"):
        empty_project.transit.export_gtfs(tmp_path / "shared.zip", URL, TIME_ZONE)

    empty_project.transit.export_gtfs(tmp_path / "export.zip", URL, TIME_ZONE, prefix_ids=True)
    exported = read_feed(tmp_path / "export.zip")
    assert {row["stop_id"] for row in exported["stops.txt"]} == {f"{a}:{stop}" for a in agencies for stop in STOPS}
    assert {row["zone_id"] for row in exported["stops.txt"]} == {f"{a}:{z}" for a in agencies for z in ["Z1", "Z2"]}
    assert {row["route_id"]: row["agency_id"] for row in exported["routes.txt"]}["SUR:N7"] == "SUR"
    stop_times = exported["stop_times.txt"]
    assert len(stop_times) == 2 * len(feed()["stop_times.txt"])
    assert all(row["trip_id"].split(":")[0] == row["stop_id"].split(":")[0] for row in stop_times)
    rules = {(row["fare_id"], row["route_id"], row["destination_id"]) for row in exported["fare_rules.txt"]}
    assert {("NORTE:one", "NORTE:R1", ""), ("SUR:pass", "", "SUR:Z2")} <= rules
