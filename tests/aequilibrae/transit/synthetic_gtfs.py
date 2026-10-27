"""Small synthetic GTFS feeds for GTFS import and export tests."""

import csv
import io
import zipfile
from itertools import accumulate

SERVICE_DATE = "2026-01-05"


def hms(seconds: int) -> str:
    """Format seconds after midnight as a GTFS time, which may exceed 24:00:00."""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def base_tables() -> dict[str, list[dict]]:
    """Return one agency, a calendar running every day of 2026 and empty service tables."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    agency = {"agency_id": "SYN", "agency_name": "SYN", "agency_url": "https://example.com", "agency_timezone": "UTC"}
    calendar = {"service_id": "all", **dict.fromkeys(days, 1), "start_date": "20260101", "end_date": "20261231"}
    return {"agency.txt": [agency], "calendar.txt": [calendar]} | {
        name: [] for name in ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
    }


def add_stop(tables: dict, stop_id: str, lat: float, lon: float, **columns) -> None:
    """Add a stop, with any further ``stops.txt`` columns, such as ``zone_id``."""
    tables["stops.txt"].append({"stop_id": stop_id, "stop_name": stop_id, "stop_lat": lat, "stop_lon": lon, **columns})


def add_trip(tables: dict, route_id: str, trip_id: str, events: list[tuple[str, int, int]], route_type: int = 3):
    """Add a trip from ``(stop_id, arrival, departure)`` events, in seconds, and its route if new."""
    if not any(route["route_id"] == route_id for route in tables["routes.txt"]):
        route = {"route_id": route_id, "route_short_name": route_id, "route_long_name": route_id}
        tables["routes.txt"].append(route | {"route_type": route_type})
    tables["trips.txt"].append({"route_id": route_id, "service_id": "all", "trip_id": trip_id, "direction_id": 0})
    for seq, (stop, arrival, departure) in enumerate(events, start=1):
        row = {"arrival_time": hms(arrival), "departure_time": hms(departure), "stop_id": stop, "stop_sequence": seq}
        tables["stop_times.txt"].append({"trip_id": trip_id} | row)


def add_trips(tables: dict, route_id: str, stops: list[str], run_minutes: list[float], departures: list[int], **kw):
    """Add a trip of a route leaving its first stop at each of ``departures``, in seconds, without dwell."""
    first = sum(trip["route_id"] == route_id for trip in tables["trips.txt"])
    for k, start in enumerate(departures, start=first):
        times = accumulate([start] + [round(minutes * 60) for minutes in run_minutes])
        add_trip(tables, route_id, f"{route_id}_{k}", [(s, t, t) for s, t in zip(stops, times, strict=True)], **kw)


def write_feed(path, tables: dict[str, list[dict]]) -> None:
    """Write each table to a zipped GTFS feed, with the union of its rows' columns."""
    with zipfile.ZipFile(path, "w") as feed:
        for name, rows in tables.items():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(c for row in rows for c in row)))
            writer.writeheader()
            writer.writerows(rows)
            feed.writestr(name, buffer.getvalue())


def import_zip(project, path, service_date: str = SERVICE_DATE, **builder_calls):
    """Import the service on ``service_date`` of a zipped GTFS feed into the project, and return the builder.

    Each keyword names a ``GTFSRouteSystemBuilder`` method called with its value before loading.
    """
    gtfs = project.scenario.create_transit_database().new_gtfs_builder(agency="SYN", file_path=str(path))
    for method, value in builder_calls.items():
        getattr(gtfs, method)(value)
    gtfs.load_date(service_date)
    gtfs.save_to_disk()
    return gtfs


def import_feed(project, tables: dict[str, list[dict]], path, **builder_calls):
    """Write a feed and import its service on ``SERVICE_DATE`` into the project, and return the builder."""
    write_feed(path, tables)
    return import_zip(project, path, **builder_calls)


def read_feed(path) -> dict[str, list[dict]]:
    """Read every file of a zipped GTFS feed as rows of text fields, keyed by file name."""
    with zipfile.ZipFile(path) as feed:
        return {name: list(csv.DictReader(io.TextIOWrapper(feed.open(name), "utf-8-sig"))) for name in feed.namelist()}


def internal_stops(project) -> dict[str, str]:
    """Map the stop IDs of the feed to internal stop IDs."""
    with project.transit_connection as conn:
        return {stop: str(stop_id) for stop_id, stop in conn.execute("SELECT stop_id, stop FROM stops").fetchall()}


def schedules(project) -> dict[str, list[tuple[int, int]]]:
    """Return each imported trip's ``(arrival, departure)`` pairs, keyed by its GTFS trip ID."""
    sql = "SELECT t.trip, s.arrival, s.departure FROM trips t JOIN trips_schedule s USING (trip_id) ORDER BY s.seq"
    out: dict[str, list[tuple[int, int]]] = {}
    with project.transit_connection as conn:
        for trip, arrival, departure in conn.execute(sql).fetchall():
            out.setdefault(trip, []).append((arrival, departure))
    return out
