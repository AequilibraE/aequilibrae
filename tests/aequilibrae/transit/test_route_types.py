import pytest

from aequilibrae.transit.route_types import base_route_type
from aequilibrae.transit.transit_elements import Pattern

from .synthetic_gtfs import add_stop, add_trips, base_tables, import_feed


@pytest.mark.parametrize("route_type, basic", [(109, 2), (200, 3), (401, 1), (405, 12), (700, 3), (800, 11), (3, 3)])
def test_extended_route_types_map_to_their_basic_type(route_type, basic):
    """Test that extended route types map onto their basic GTFS route type, and basic types onto themselves."""
    assert base_route_type(route_type) == basic


@pytest.fixture
def gtfs(empty_project, tmp_path):
    """Import a feed with a route of a basic bus, an extended bus, suburban rail and a taxi type."""
    tables = base_tables()
    for i, stop in enumerate("ABC"):
        add_stop(tables, stop, -29.95, -71.33 + i * 0.004)
    for route_id, route_type in [("bus", 3), ("xbus", 700), ("rail", 109), ("taxi", 1500)]:
        add_trips(tables, route_id, list("ABC"), [3, 4], [7 * 3600], route_type=route_type)
    return import_feed(empty_project, tables, tmp_path / "feed.zip")


def test_extended_route_types_get_the_defaults_of_their_basic_type(empty_project, gtfs):
    """Test that imported extended route types take the capacities and PCEs of their basic type."""
    with empty_project.transit_connection as conn:
        routes = {r[0]: r[1:] for r in conn.execute("SELECT route, seated_capacity, total_capacity, pce FROM routes")}
    assert routes == {"bus": (30, 60, 4), "xbus": (30, 60, 4), "rail": (700, 700, 2), "taxi": (30, 60, 2)}


def test_map_match_selects_extended_types_through_their_basic_type(gtfs, monkeypatch):
    """Test that map matching bus routes also selects extended bus types."""
    matched = []
    monkeypatch.setattr(Pattern, "map_match", lambda pattern: matched.append(pattern.route_type))
    gtfs.map_match()
    assert sorted(matched) == [3, 700]
