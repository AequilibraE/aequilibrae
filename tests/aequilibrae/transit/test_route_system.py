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
    assert route_system.routes["route_id"].is_unique
    assert route_system.trips["trip_id"].is_unique

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
            routes = pd.read_csv(routes_file, sep=",")
        with archive.open("stops.txt") as stops_file:
            stops = pd.read_csv(stops_file, sep=",")
        with archive.open("shapes.txt") as shapes_file:
            shapes = pd.read_csv(shapes_file, sep=",")
        with archive.open("trips.txt") as trips_file:
            trips = pd.read_csv(trips_file, sep=",")
        with archive.open("stop_times.txt") as stop_times_file:
            stop_times = pd.read_csv(stop_times_file, sep=",")

    assert not routes.empty
    assert not stops.empty
    assert not shapes.empty
    assert not trips.empty
    assert not stop_times.empty
    assert routes["route_id"].is_unique
    assert trips["trip_id"].is_unique
    assert stop_times["stop_id"].notna().all()
    assert stop_times["trip_id"].isin(trips["trip_id"]).all()
    assert shapes["shape_id"].nunique() == route_system.patterns["shape_id"].nunique()
    assert stop_times.groupby("trip_id")["stop_sequence"].apply(lambda s: s.is_monotonic_increasing).all()
    assert list(stop_times.columns) == ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]

