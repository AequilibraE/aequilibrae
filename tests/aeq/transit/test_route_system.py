import zipfile
from pathlib import Path

import pandas as pd

from aequilibrae.transit.route_system import RouteSystem


def test_route_system_exports_gtfs_tables(build_gtfs_project, tmp_path):
    transit = build_gtfs_project
    builder = transit.new_gtfs_builder(
        agency="Agency_1",
        day="2016-04-13",
        file_path=Path(transit.project.project_base_path) / "gtfs_coquimbo.zip",
    )
    builder.load_date("2016-04-13")
    builder.save_to_disk()

    route_system = RouteSystem(transit.project)
    route_system.load_route_system()

    assert isinstance(route_system.agencies, pd.DataFrame)
    assert isinstance(route_system.stops, pd.DataFrame)
    assert isinstance(route_system.routes, pd.DataFrame)
    assert isinstance(route_system.trips, pd.DataFrame)
    assert isinstance(route_system.patterns, pd.DataFrame)
    assert isinstance(route_system.stop_times, pd.DataFrame)
    assert not route_system.routes.empty
    assert not route_system.trips.empty
    assert not route_system.stop_times.empty

    route_system.write_GTFS(str(tmp_path))

    output_zip = tmp_path / "aequilibrae_gtfs.zip"
    assert output_zip.exists()

    with zipfile.ZipFile(output_zip) as archive:
        names = sorted(archive.namelist())
        assert names == sorted(
            [
                "agency.txt",
                "calendar.txt",
                "fare_attributes.txt",
                "fare_rules.txt",
                "routes.txt",
                "shapes.txt",
                "stop_times.txt",
                "stops.txt",
                "trips.txt",
            ]
        )

        with archive.open("routes.txt") as routes_file:
            routes = pd.read_csv(routes_file)
        with archive.open("trips.txt") as trips_file:
            trips = pd.read_csv(trips_file)
        with archive.open("stop_times.txt") as stop_times_file:
            stop_times = pd.read_csv(stop_times_file)

    assert not routes.empty
    assert not trips.empty
    assert not stop_times.empty
    assert list(stop_times.columns) == ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]

