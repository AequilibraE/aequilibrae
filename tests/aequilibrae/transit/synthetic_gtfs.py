"""Small synthetic GTFS feeds for GTFS import tests."""

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


def add_route(tables: dict, route_id: str, route_type: int = 3) -> None:
    """Add a route, unless the feed already has one with this ID."""
    if not any(route["route_id"] == route_id for route in tables["routes.txt"]):
        tables["routes.txt"].append(
            {"route_id": route_id, "route_short_name": route_id, "route_long_name": route_id, "route_type": route_type}
        )


def add_trip(tables: dict, route_id: str, trip_id: str, events: list[tuple[str, int, int]], route_type: int = 3):
    """Add a trip from ``(stop_id, arrival, departure)`` events, in seconds, in stop sequence order."""
    add_route(tables, route_id, route_type)
    tables["trips.txt"].append({"route_id": route_id, "service_id": "all", "trip_id": trip_id, "direction_id": 0})
    for seq, (stop, arrival, departure) in enumerate(events, start=1):
        tables["stop_times.txt"].append(
            {
                "trip_id": trip_id,
                "arrival_time": hms(arrival),
                "departure_time": hms(departure),
                "stop_id": stop,
                "stop_sequence": seq,
            }
        )


def add_trips(
    tables: dict,
    route_id: str,
    stops: list[str],
    run_minutes: list[float],
    departures: list[int],
    route_type: int = 3,
) -> None:
    """Add a route with one trip leaving its first stop at each of ``departures``, in seconds.

    Arrival and departure are equal at every stop, so scheduled dwell is zero.
    """
    if len(run_minutes) != len(stops) - 1:
        raise ValueError("Each pair of consecutive stops needs one run time")

    first = sum(trip["route_id"] == route_id for trip in tables["trips.txt"])
    for k, start in enumerate(departures, start=first):
        times = [start]
        for minutes in run_minutes:
            times.append(times[-1] + round(minutes * 60))
        events = [(stop, time, time) for stop, time in zip(stops, times, strict=True)]
        add_trip(tables, route_id, f"{route_id}_{k}", events, route_type)


def add_frequency(tables: dict, trip_id: str, start: int, end: int, headway: int) -> None:
    """Add a ``frequencies.txt`` window repeating a trip every ``headway`` from ``start`` until ``end``, in seconds."""
    tables.setdefault("frequencies.txt", []).append(
        {"trip_id": trip_id, "start_time": hms(start), "end_time": hms(end), "headway_secs": headway}
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


def import_feed(project, tables: dict[str, list[dict]], path: Path, agency: str = "SYN", **builder_calls):
    """Write a feed and import its service on ``SERVICE_DATE`` into the project's transit database.

    Each keyword names a ``GTFSRouteSystemBuilder`` method called with its value before loading, such as
    ``set_maximum_speeds=df``. Returns the builder.
    """
    feed = write_feed(path, tables)
    transit = project.scenario.create_transit_database()
    gtfs = transit.new_gtfs_builder(agency=agency, file_path=str(feed))
    for method, value in builder_calls.items():
        getattr(gtfs, method)(value)
    gtfs.load_date(SERVICE_DATE)
    gtfs.save_to_disk()
    return gtfs


def internal_stops(project) -> dict[str, str]:
    """Map the stop IDs of the feed to internal stop IDs."""
    with project.transit_connection as conn:
        return {stop: str(stop_id) for stop_id, stop in conn.execute("SELECT stop_id, stop FROM stops").fetchall()}


def schedules(project) -> dict[str, list[tuple[int, int]]]:
    """Return each imported trip's ``(arrival, departure)`` pairs, keyed by its GTFS trip ID."""
    sql = """SELECT t.trip, s.arrival, s.departure FROM trips t JOIN trips_schedule s ON t.trip_id = s.trip_id
             ORDER BY t.trip, s.seq"""
    out: dict[str, list[tuple[int, int]]] = {}
    with project.transit_connection as conn:
        for trip, arrival, departure in conn.execute(sql).fetchall():
            out.setdefault(trip, []).append((arrival, departure))
    return out


def import_zip(project, path: Path, agency: str = "SYN", service_date: str = SERVICE_DATE):
    """Import the service on ``service_date`` of an existing zipped GTFS feed into the project. Returns the builder.

    The project must be the active one, which the builder imports into.
    """
    transit = project.scenario.create_transit_database()
    gtfs = transit.new_gtfs_builder(agency=agency, file_path=str(path))
    gtfs.load_date(service_date)
    gtfs.save_to_disk()
    return gtfs


def read_feed(path: Path) -> dict[str, list[dict]]:
    """Read every file of a zipped GTFS feed as rows of text fields, keyed by file name."""
    with zipfile.ZipFile(path) as feed:
        return {
            name: list(csv.DictReader(io.TextIOWrapper(feed.open(name), encoding="utf-8-sig")))
            for name in feed.namelist()
        }
