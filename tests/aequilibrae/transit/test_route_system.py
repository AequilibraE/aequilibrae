import zipfile
from pathlib import Path

import pandas as pd
import pytest

def test_transit_exports_gtfs_tables(build_gtfs_project, tmp_path):
    transit = build_gtfs_project
    builder = transit.new_gtfs_builder(
        agency="Agency_1",
        day="2016-04-13",
        file_path=Path(transit.project.project_base_path) / "gtfs_coquimbo.zip",
    )
    builder.load_date("2016-04-13")
    builder.save_to_disk()

    with transit.project.transit_connection as conn:
        pattern_count = conn.execute(
            "SELECT COUNT(DISTINCT pattern_id) FROM routes WHERE geometry IS NOT NULL"
        ).fetchone()[0]
        route_count = conn.execute("SELECT COUNT(DISTINCT route_id) FROM routes").fetchone()[0]

    transit.export_gtfs(str(tmp_path))

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
    assert routes["route_id"].nunique() == route_count
    assert trips["trip_id"].is_unique
    assert stop_times["stop_id"].notna().all()
    assert stop_times["trip_id"].isin(trips["trip_id"]).all()
    assert trips["route_id"].isin(routes["route_id"]).all()
    assert shapes["shape_id"].nunique() == pattern_count
    assert stop_times.groupby("trip_id")["stop_sequence"].apply(lambda s: s.is_monotonic_increasing).all()
    assert list(stop_times.columns) == ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]


def test_transit_export_gtfs_rejects_conflicting_pattern_route_metadata(build_gtfs_project, tmp_path):
    transit = build_gtfs_project
    builder = transit.new_gtfs_builder(
        agency="Agency_1",
        day="2016-04-13",
        file_path=Path(transit.project.project_base_path) / "gtfs_coquimbo.zip",
    )
    builder.load_date("2016-04-13")
    builder.save_to_disk()

    with transit.project.transit_connection as conn:
        route_id = conn.execute(
            "SELECT route_id FROM routes GROUP BY route_id HAVING COUNT(*) > 1 ORDER BY route_id LIMIT 1"
        ).fetchone()[0]
        pattern_ids = [
            row[0]
            for row in conn.execute(
                "SELECT pattern_id FROM routes WHERE route_id=? ORDER BY pattern_id LIMIT 2", [route_id]
            ).fetchall()
        ]
        conn.execute("UPDATE routes SET longname=? WHERE pattern_id=?", ["conflict A", pattern_ids[0]])
        conn.execute("UPDATE routes SET longname=? WHERE pattern_id=?", ["conflict B", pattern_ids[1]])

    message = f"Cannot export GTFS route_id {route_id}: conflicting values for route_long_name"
    with pytest.raises(ValueError, match=message):
        transit.export_gtfs(str(tmp_path))


