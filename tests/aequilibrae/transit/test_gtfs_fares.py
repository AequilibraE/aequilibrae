from .synthetic_gtfs import add_stop, add_trips, base_tables, import_feed

FARE = ["fare_id", "price", "currency_type", "payment_method", "transfers", "transfer_duration"]
RULE = ["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]


def fare_feed() -> dict:
    """Return a route over stops in alphanumeric fare zones, with free, unlimited and limited transfer fares."""
    tables = base_tables()
    for i, zone in enumerate(["Z1", "07", "CENTRO"]):
        add_stop(tables, "ABC"[i], -29.95, -71.33 + i * 0.004, zone_id=zone)
    add_trips(tables, "R1", list("ABC"), [3, 4], [7 * 3600])
    fares = [["free", 0, "CLP", 0, 0, ""], ["unlimited", 650, "CLP", "", "", ""], ["two", 750.5, "CLP", 1, 2, 5400]]
    rules = [["free", "R1", "", "", ""], ["unlimited", "", "Z1", "07", "CENTRO"], ["two", "", "", "", "07"]]
    tables["fare_attributes.txt"] = [dict(zip(FARE, row, strict=True)) for row in fares]
    tables["fare_rules.txt"] = [dict(zip(RULE, row, strict=True)) for row in rules]
    return tables


def fetch(project, sql: str) -> list[tuple]:
    with project.transit_connection as conn:
        return sorted(conn.execute(sql).fetchall(), key=str)


def test_fare_attributes_are_stored_as_published(empty_project, tmp_path):
    """Test that free fares keep their zero price and empty transfers, durations and payment methods are NULL."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    sql = "SELECT fare, price, payment_method, transfer, transfer_duration FROM fare_attributes"
    assert fetch(empty_project, sql) == [
        ("free", 0, 0, 0, None),
        ("two", 750.5, 1, 2, 5400),
        ("unlimited", 650, None, None, None),
    ]


def test_fare_zones_are_stored_as_text(empty_project, tmp_path):
    """Test that fare zone IDs of stops and fare rules are stored as published, as text."""
    import_feed(empty_project, fare_feed(), tmp_path / "feed.zip")
    assert fetch(empty_project, "SELECT stop, transit_fare_zone FROM stops") == [
        ("A", "Z1"),
        ("B", "07"),
        ("C", "CENTRO"),
    ]
    sql = """SELECT a.fare, r.route_id IS NOT NULL, r.origin, r.destination, r."contains"
             FROM fare_rules r JOIN fare_attributes a USING (fare_id)"""
    assert fetch(empty_project, sql) == [
        ("free", 1, None, None, None),
        ("two", 0, None, None, "07"),
        ("unlimited", 0, "Z1", "07", "CENTRO"),
    ]
    assert fetch(empty_project, "SELECT transit_fare_zone FROM fare_zones") == [("07",), ("Z1",)]


def test_fare_rules_for_unknown_routes_or_fares_are_skipped(empty_project, tmp_path):
    """Test that fare rules for a route or a fare absent from the feed are skipped, not applied to every route."""
    tables = fare_feed()
    unknown = [["free", "R9", "", "", ""], ["gone", "R1", "", "", ""]]
    tables["fare_rules.txt"] += [dict(zip(RULE, row, strict=True)) for row in unknown]
    import_feed(empty_project, tables, tmp_path / "feed.zip")
    sql = "SELECT a.fare, r.route_id IS NOT NULL FROM fare_rules r JOIN fare_attributes a USING (fare_id)"
    assert fetch(empty_project, sql) == [("free", 1), ("two", 0), ("unlimited", 0)]
