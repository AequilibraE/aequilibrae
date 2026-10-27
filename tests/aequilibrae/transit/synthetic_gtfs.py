"""Small synthetic GTFS feeds for transit reference tests."""

import csv
import io
import zipfile
from pathlib import Path

SERVICE_DATE = "2026-01-05"


def hms(seconds: int) -> str:
    """Format seconds after midnight as a GTFS time, which may exceed 24:00:00."""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def base_tables(agency_id: str = "SYN") -> dict[str, list[dict]]:
    """Return one agency, a calendar running every day of 2026 and empty service tables."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    return {
        "agency.txt": [
            {
                "agency_id": agency_id,
                "agency_name": agency_id,
                "agency_url": "https://example.com",
                "agency_timezone": "UTC",
            }
        ],
        "calendar.txt": [
            {"service_id": "all", **dict.fromkeys(days, 1), "start_date": "20260101", "end_date": "20261231"}
        ],
        "stops.txt": [],
        "routes.txt": [],
        "trips.txt": [],
        "stop_times.txt": [],
    }


def add_stop(tables: dict, stop_id: str, lat: float, lon: float, **columns) -> None:
    """Add a stop, with any further ``stops.txt`` columns, such as ``parent_station``."""
    tables["stops.txt"].append({"stop_id": stop_id, "stop_name": stop_id, "stop_lat": lat, "stop_lon": lon, **columns})


def add_timetable(
    tables: dict,
    route_id: str,
    stops: list[str],
    run_minutes: list[float],
    headway_minutes: float,
    first: int,
    last: int,
    route_type: int = 3,
) -> None:
    """Add a route with one trip every headway, departing from ``first`` up to but excluding ``last``.

    Arrival and departure are equal at every stop, so scheduled dwell is zero.
    """
    if len(run_minutes) != len(stops) - 1:
        raise ValueError("Each pair of consecutive stops needs one run time")

    tables["routes.txt"].append(
        {"route_id": route_id, "route_short_name": route_id, "route_long_name": route_id, "route_type": route_type}
    )
    for k, start in enumerate(range(first, last, round(headway_minutes * 60))):
        trip_id = f"{route_id}_{k}"
        tables["trips.txt"].append({"route_id": route_id, "service_id": "all", "trip_id": trip_id, "direction_id": 0})
        time = start
        for seq, stop in enumerate(stops):
            if seq:
                time += round(run_minutes[seq - 1] * 60)
            tables["stop_times.txt"].append(
                {
                    "trip_id": trip_id,
                    "arrival_time": hms(time),
                    "departure_time": hms(time),
                    "stop_id": stop,
                    "stop_sequence": seq + 1,
                }
            )


def write_feed(path: Path, tables: dict[str, list[dict]]) -> Path:
    """Write each table to a zipped GTFS feed, with the union of its rows' columns."""
    with zipfile.ZipFile(path, "w") as feed:
        for name, rows in tables.items():
            columns = list(dict.fromkeys(column for row in rows for column in row))
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            feed.writestr(name, buffer.getvalue())
    return path
