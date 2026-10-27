from .synthetic_gtfs import add_stop, add_trips, base_tables, import_feed, internal_stops


def fare(fare_id, price, transfers, transfer_duration="", payment_method=0) -> dict:
    return {
        "fare_id": fare_id,
        "price": price,
        "currency_type": "CLP",
        "payment_method": payment_method,
        "transfers": transfers,
        "transfer_duration": transfer_duration,
    }


def rule(fare_id, route_id="", origin_id="", destination_id="", contains_id="") -> dict:
    return {
        "fare_id": fare_id,
        "route_id": route_id,
        "origin_id": origin_id,
        "destination_id": destination_id,
        "contains_id": contains_id,
    }


def fare_feed() -> dict:
    """Return a feed with one route over four stops in three alphanumeric fare zones, and Fares v1 files."""
    tables = base_tables()
    for i, (stop, zone) in enumerate(zip("ABCD", ["Z1", "Z2", "CENTRO", ""], strict=True)):
        add_stop(tables, stop, -29.95, -71.33 + i * 0.004, zone_id=zone)
    add_trips(tables, "R1", list("ABCD"), [3, 4, 3], [7 * 3600])
    tables["fare_attributes.txt"] = [
        fare("free", 0, 0),
        fare("unlimited", 650, ""),
        fare("one", 700, 1, 3600, payment_method=1),
        fare("two", 750.5, 2, 5400),
    ]
    tables["fare_rules.txt"] = [
        rule("one", route_id="R1"),
        rule("two", origin_id="Z1", destination_id="Z2"),
        rule("unlimited", contains_id="CENTRO"),
        rule("free", route_id="R1", origin_id="Z2", destination_id="CENTRO", contains_id="Z1"),
    ]
    return tables


def fetch(project, sql: str) -> list[tuple]:
    with project.transit_connection as conn:
        return conn.execute(sql).fetchall()


def fare_attributes(project) -> dict[str, tuple]:
    sql = "SELECT fare, price, currency, payment_method, transfer, transfer_duration FROM fare_attributes"
    return {row[0]: row[1:] for row in fetch(project, sql)}


def fare_rules(project) -> dict[str, tuple]:
    sql = """SELECT a.fare, r.route_id, r.origin, r.destination, r.contains
             FROM fare_rules r JOIN fare_attributes a ON r.fare_id = a.fare_id"""
    return {row[0]: row[1:] for row in fetch(project, sql)}


def test_free_fare_is_stored_with_a_zero_price(empty_project, tmp_path):
    """Test that a free fare keeps its price of zero, and its zero transfers, instead of becoming NULL."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    price, currency, payment_method, transfer, _ = fare_attributes(empty_project)["free"]
    assert (price, currency, payment_method, transfer) == (0, "CLP", 0, 0)
    assert None not in (price, payment_method, transfer)


def test_transfers_are_stored_as_published(empty_project, tmp_path):
    """Test that empty transfers (unlimited) are stored as NULL and 0, 1 and 2 transfers as given."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    fares = fare_attributes(empty_project)
    assert {fare_id: fares[fare_id][3] for fare_id in fares} == {"free": 0, "unlimited": None, "one": 1, "two": 2}
    assert fares["one"][:3] == (700, "CLP", 1)
    assert fares["two"][0] == 750.5


def test_transfer_durations_are_stored_as_published(empty_project, tmp_path):
    """Test that an empty transfer_duration (no limit) is stored as NULL and given durations are kept."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    fares = fare_attributes(empty_project)
    assert {fare_id: fares[fare_id][4] for fare_id in fares} == {
        "free": None,
        "unlimited": None,
        "one": 3600,
        "two": 5400,
    }


def test_alphanumeric_fare_zones_are_stored_as_text(empty_project, tmp_path):
    """Test that alphanumeric zone IDs of stops and fare rules are stored as text, each with a fare_zones row."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    stops = internal_stops(empty_project)

    zones = dict(fetch(empty_project, "SELECT stop_id, transit_fare_zone FROM stops"))
    assert {stop: zones[stop_id] for stop, stop_id in stops.items()} == {"A": "Z1", "B": "Z2", "C": "CENTRO", "D": None}

    rules = fare_rules(empty_project)
    assert rules["two"][1:] == ("Z1", "Z2", None)
    assert rules["unlimited"] == (None, None, None, "CENTRO")
    assert rules["free"][1:] == ("Z2", "CENTRO", "Z1")

    (agency_id,) = fetch(empty_project, "SELECT agency_id FROM agencies")[0]
    fare_zones = fetch(empty_project, "SELECT transit_fare_zone, agency_id FROM fare_zones")
    assert sorted(fare_zones) == [("CENTRO", agency_id), ("Z1", agency_id), ("Z2", agency_id)]


def test_numeric_looking_fare_zones_keep_their_leading_zeros(empty_project, tmp_path):
    """Test that zone IDs that look like numbers, such as 07, are stored as published in stops and fare rules."""
    tables = fare_feed()
    tables["stops.txt"][0]["zone_id"] = "07"
    tables["fare_rules.txt"] = [rule("two", origin_id="07", destination_id="Z2", contains_id="07")]
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert fare_rules(empty_project)["two"][1:] == ("07", "Z2", "07")
    zones = {zone for (zone,) in fetch(empty_project, "SELECT transit_fare_zone FROM stops")}
    assert "07" in zones


def test_fare_rule_without_zones_applies_to_its_route(empty_project, tmp_path):
    """Test that a fare rule giving only a route is stored with that route and without any zone."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    (route_id,) = fetch(empty_project, "SELECT route_id FROM routes WHERE route='R1'")[0]
    rules = fare_rules(empty_project)
    assert rules["one"] == (route_id, None, None, None)
    assert rules["free"][0] == route_id


def test_fare_rule_for_an_unknown_route_is_skipped(empty_project, tmp_path):
    """Test that a fare rule for a route absent from routes.txt is skipped instead of applying to every route."""
    tables = fare_feed()
    tables["fare_rules.txt"] = [rule("one", route_id="R1"), rule("two", route_id="R9")]
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    assert list(fare_rules(empty_project)) == ["one"]


def test_feed_without_fare_files_imports(empty_project, tmp_path):
    """Test that a feed without fare_attributes.txt and fare_rules.txt imports, leaving the fare tables empty."""
    tables = fare_feed()
    del tables["fare_attributes.txt"], tables["fare_rules.txt"]
    import_feed(empty_project, tables, tmp_path / "feed.zip")

    assert len(internal_stops(empty_project)) == 4
    for table in ["fare_attributes", "fare_rules"]:
        assert fetch(empty_project, f"SELECT COUNT(*) FROM {table}")[0][0] == 0
    # The fare zones of the stops are still recorded
    assert sorted(fetch(empty_project, "SELECT transit_fare_zone FROM fare_zones")) == [("CENTRO",), ("Z1",), ("Z2",)]
