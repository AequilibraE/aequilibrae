from os.path import join

import pytest

from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import read_and_close


@pytest.fixture(scope="function")
def create_project(empty_project: Project):
    yield Transit(empty_project)


@pytest.mark.parametrize(
    "table, exp_column",
    [
        ("agencies", ["agency_id", "agency", "feed_date", "service_date", "description"]),
        (
            "fare_attributes",
            ["fare_id", "fare", "agency_id", "price", "currency", "payment_method", "transfer", "transfer_duration"],
        ),
        ("fare_rules", ["fare_id", "route_id", "origin", "destination", "contains"]),
        ("fare_zones", ["transit_fare_zone", "agency_id"]),
        ("pattern_mapping", ["pattern_id", "seq", "link", "dir", "geometry"]),
        (
            "routes",
            [
                "pattern_id",
                "route_id",
                "route",
                "agency_id",
                "shortname",
                "longname",
                "description",
                "route_type",
                "pce",
                "seated_capacity",
                "total_capacity",
                "geometry",
            ],
        ),
        ("route_links", ["transit_link", "pattern_id", "seq", "from_stop", "to_stop", "distance", "geometry"]),
        (
            "stops",
            [
                "stop_id",
                "stop",
                "agency_id",
                "link",
                "dir",
                "name",
                "parent_station",
                "description",
                "street",
                "zone_id",
                "transit_fare_zone",
                "route_type",
                "geometry",
            ],
        ),
        ("trips", ["trip_id", "trip", "dir", "pattern_id"]),
        ("trips_schedule", ["trip_id", "seq", "arrival", "departure"]),
    ],
)
def test_create_table(table: str, exp_column: list, create_project):
    with read_and_close(join(create_project.project_base_path, "public_transport.sqlite")) as conn:
        fields = [x[1] for x in conn.execute(f"PRAGMA table_info({table});").fetchall()]

    assert exp_column == fields, f"Table {table.upper()} was not created correctly"
