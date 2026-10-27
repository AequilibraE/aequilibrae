import pytest

from aequilibrae.transit.route_types import base_route_type, lookup_by_route_type
from aequilibrae.transit.transit_elements import Pattern, mode_corresp

from .synthetic_gtfs import add_stop, add_trips, base_tables, import_feed


@pytest.mark.parametrize(
    "route_type, expected",
    [
        (100, 2),
        (109, 2),
        (117, 2),
        (200, 3),
        (400, 1),
        (401, 1),
        (405, 12),
        (700, 3),
        (716, 3),
        (800, 11),
        (900, 0),
        (1000, 4),
        (1200, 4),
        (1300, 6),
        (1400, 7),
    ],
)
def test_extended_route_types_map_to_their_basic_type(route_type, expected):
    """Test that each extended route type family maps onto its basic GTFS route type."""
    assert base_route_type(route_type) == expected


@pytest.mark.parametrize("route_type", [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 1100, 1500, 1700])
def test_basic_and_unmapped_route_types_are_unchanged(route_type):
    """Test that basic route types and extended types without a basic equivalent are returned unchanged."""
    assert base_route_type(route_type) == route_type


def test_lookup_prefers_the_route_type_then_its_basic_type():
    """Test that a lookup uses the route type's own entry, then its basic type's entry, then the default."""
    table = {3: "bus", 704: "local bus"}
    assert lookup_by_route_type(table, 704) == "local bus"
    assert lookup_by_route_type(table, 700) == "bus"
    assert lookup_by_route_type(table, 109, "other") == "other"


def test_extended_bus_and_trolleybus_types_have_a_road_mode():
    """Test that extended bus and trolleybus types map onto the road mode used for map matching."""
    assert lookup_by_route_type(mode_corresp, 700) == mode_corresp[3]
    assert lookup_by_route_type(mode_corresp, 800) == mode_corresp[11]
    assert lookup_by_route_type(mode_corresp, 109) is None


def extended_type_feed():
    """Return a feed with one route for each of a basic bus, an extended bus, suburban rail and a taxi type."""
    tables = base_tables()
    for i, stop in enumerate("ABC"):
        add_stop(tables, stop, -29.95, -71.33 + i * 0.004)
    for route_id, route_type in [("bus", 3), ("xbus", 700), ("rail", 109), ("taxi", 1500)]:
        add_trips(tables, route_id, list("ABC"), [3, 4], [7 * 3600], route_type=route_type)
    return tables


def test_extended_route_types_get_the_defaults_of_their_basic_type(empty_project, tmp_path):
    """Test that imported extended route types take the capacities and PCEs of their basic type."""
    import_feed(empty_project, extended_type_feed(), tmp_path / "feed.zip")

    sql = "SELECT route, route_type, seated_capacity, total_capacity, pce FROM routes"
    with empty_project.transit_connection as conn:
        routes = {row[0]: row[1:] for row in conn.execute(sql).fetchall()}

    assert routes["bus"] == (3, 30, 60, 4.0)
    assert routes["xbus"] == (700, 30, 60, 4.0)
    assert routes["rail"] == (109, 700, 700, 2.0)
    assert routes["taxi"] == (1500, 30, 60, 2.0)


def test_extended_route_type_entries_override_their_basic_type(empty_project, tmp_path):
    """Test that capacities and PCEs given for an extended route type override those of its basic type."""
    capacities = {3: [30, 60], 700: [50, 90], "other": [10, 20]}
    pces = {3: 4.0, 700: 5.5, "other": 1.0}
    import_feed(empty_project, extended_type_feed(), tmp_path / "feed.zip", set_capacities=capacities, set_pces=pces)

    sql = "SELECT route, seated_capacity, total_capacity, pce FROM routes"
    with empty_project.transit_connection as conn:
        routes = {row[0]: row[1:] for row in conn.execute(sql).fetchall()}

    assert routes["bus"] == (30, 60, 4.0)
    assert routes["xbus"] == (50, 90, 5.5)
    assert routes["taxi"] == (10, 20, 1.0)


def test_map_match_selects_extended_types_through_their_basic_type(empty_project, tmp_path, monkeypatch, caplog):
    """Test that map matching bus routes also selects extended bus types, and extended types are accepted."""
    matched = []
    monkeypatch.setattr(Pattern, "map_match", lambda pattern: matched.append(pattern.route_type))
    gtfs = import_feed(empty_project, extended_type_feed(), tmp_path / "feed.zip")

    gtfs.map_match()
    assert sorted(matched) == [3, 700]

    matched.clear()
    with caplog.at_level("WARNING"):
        gtfs.map_match(route_types=(700,))
    assert matched == [700]
    assert "no corresponding road mode" not in caplog.text
