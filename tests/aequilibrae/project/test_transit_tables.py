from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
import pytest


class TestTransitTables:
    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("agencies", ["agency_id", "agency", "feed_date", "service_date", "description"]),
        ],
    )
    def test_create_agencies_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            (
                "fare_attributes",
                [
                    "fare_id",
                    "fare",
                    "agency_id",
                    "price",
                    "currency",
                    "payment_method",
                    "transfer",
                    "transfer_duration",
                ],
            ),
        ],
    )
    def test_create_fare_attributes_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("fare_rules", ["fare_id", "route_id", "origin", "destination", "contains"]),
        ],
    )
    def test_create_fare_rules_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("route_links", ["transit_link", "pattern_id", "seq", "from_stop", "to_stop", "distance", "geometry"]),
        ],
    )
    def test_create_route_links_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("fare_zones", ["fare_zone_id", "transit_zone", "agency_id"]),
        ],
    )
    def test_create_fare_zones_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("trips", ["trip_id", "trip", "dir", "pattern_id"]),
        ],
    )
    def test_create_trips_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("trips_schedule", ["trip_id", "seq", "arrival", "departure"]),
        ],
    )
    def test_create_trips_schedule_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
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
                    "fare_zone_id",
                    "route_type",
                    "geometry",
                ],
            ),
        ],
    )
    def test_create_stops_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("pattern_mapping", ["pattern_id", "seq", "link", "dir", "geometry"]),
        ],
    )
    def test_create_pattern_mapping_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
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
                    "seated_capacity",
                    "total_capacity",
                    "geometry",
                ],
            ),
        ],
    )
    def test_create_routes_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"

    @pytest.mark.parametrize(
        "table, exp_column",
        [
            ("stop_connectors", ["id_from", "id_to", "conn_type", "traversal_time", "penalty_cost", "geometry"]),
        ],
    )
    def test_create_stop_connectors_table(self, table: str, exp_column: list, project: Project):

        curr = database_connection(table_type="transit").cursor()
        curr.execute(f"PRAGMA table_info({table});")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        assert exp_column == fields, f"Table {table.upper()} was not created correctly"
