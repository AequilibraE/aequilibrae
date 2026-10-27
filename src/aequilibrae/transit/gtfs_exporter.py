"""Builds a GTFS feed of the transit services stored in a transit database, for ``Transit.export_gtfs``.

IDs are written as published in the GTFS feeds imported (stops, routes, trips, fares and fare zones), so a feed
exported and imported again keeps them. What the database does not store (e.g. the agency URL and time zone)
is either given by the caller or left out.
"""

import logging
import os
import zipfile
from datetime import date
from io import TextIOWrapper
from numbers import Integral
from pathlib import Path
from sqlite3 import Connection
from zoneinfo import available_timezones

import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger(__name__)

# Files with more rows than this are written with ZIP64 records, which files over 2 GB need
ZIP64_ROWS = 2_000_000

STOPS_SQL = """SELECT CAST(stop_id AS TEXT) AS stop_id, stop, agency_id, name, description,
                      CAST(parent_station AS TEXT) AS parent_station, CAST(transit_fare_zone AS TEXT) AS zone,
                      ST_Y(geometry) AS lat, ST_X(geometry) AS lon
               FROM stops WHERE agency_id IN ({ids}) ORDER BY agency_id, stop"""

PATTERNS_SQL = """SELECT pattern_id, route_id, route, agency_id, shortname, longname, description, route_type
                  FROM routes WHERE agency_id IN ({ids}) ORDER BY pattern_id"""

TRIPS_SQL = """SELECT t.trip_id, t.trip, t.dir, t.pattern_id FROM trips t
               JOIN routes r ON r.pattern_id = t.pattern_id WHERE r.agency_id IN ({ids}) ORDER BY t.trip_id"""

LINKS_SQL = """SELECT l.pattern_id, l.seq, CAST(l.from_stop AS TEXT) AS from_stop, CAST(l.to_stop AS TEXT) AS to_stop
               FROM route_links l JOIN routes r ON r.pattern_id = l.pattern_id
               WHERE r.agency_id IN ({ids}) ORDER BY l.pattern_id, l.seq"""

SCHEDULE_SQL = """SELECT s.trip_id, s.seq, s.arrival, s.departure FROM trips_schedule s
                  JOIN trips t ON t.trip_id = s.trip_id JOIN routes r ON r.pattern_id = t.pattern_id
                  WHERE r.agency_id IN ({ids}) ORDER BY s.trip_id, s.seq"""

PATTERN_GEO_SQL = """SELECT pattern_id, ST_AsBinary(geometry) AS wkb FROM routes
                     WHERE agency_id IN ({ids}) AND geometry IS NOT NULL ORDER BY pattern_id"""

LINK_GEO_SQL = """SELECT l.pattern_id, ST_AsBinary(l.geometry) AS wkb FROM route_links l
                  JOIN routes r ON r.pattern_id = l.pattern_id
                  WHERE r.agency_id IN ({ids}) AND l.geometry IS NOT NULL ORDER BY l.pattern_id, l.seq"""

FARES_SQL = """SELECT fare_id, fare, agency_id, price, currency, payment_method, transfer, transfer_duration
               FROM fare_attributes WHERE agency_id IN ({ids}) ORDER BY fare_id"""

RULES_SQL = """SELECT r.fare_id, r.route_id, CAST(r.origin AS TEXT) AS origin,
                      CAST(r.destination AS TEXT) AS destination, CAST(r."contains" AS TEXT) AS "contains"
               FROM fare_rules r JOIN fare_attributes a ON a.fare_id = r.fare_id WHERE a.agency_id IN ({ids})"""


def gtfs_tables(
    conn: Connection,
    agency_url: str | dict,
    agency_timezone: str,
    agencies: list[int | str] | int | str | None = None,
    prefix_ids: bool = False,
) -> dict[str, pd.DataFrame]:
    """Builds the contents of each GTFS file for the transit services in a transit database.

    See ``Transit.export_gtfs`` for the arguments.

    :Arguments:
        **conn** (:obj:`sqlite3.Connection`): Spatial connection to the transit database

    :Returns:
        **tables** (:obj:`dict`): Contents of each GTFS file, keyed by file name (e.g. 'stops.txt')
    """
    agency = _agencies(conn, agencies, agency_url, agency_timezone)
    ids = agency.agency_id.tolist()
    multiple = len(ids) > 1
    names = pd.Series(agency.agency.to_numpy(), index=ids)
    prefix = names + ":" if prefix_ids else pd.Series("", index=ids)

    stops = _read(conn, STOPS_SQL, ids)
    stops[["name", "description", "parent_station", "zone"]] = stops[
        ["name", "description", "parent_station", "zone"]
    ].fillna("")
    patterns = _read(conn, PATTERNS_SQL, ids)
    patterns[["shortname", "longname", "description"]] = patterns[["shortname", "longname", "description"]].fillna("")
    trips = _trips(conn, ids, patterns)
    fares = _read(conn, FARES_SQL, ids, objects=True)
    rules = _read(conn, RULES_SQL, ids, objects=True).merge(fares[["fare_id", "agency_id"]], on="fare_id")
    zone_columns = ["origin", "destination", "contains"]
    for column in zone_columns:
        rules[column] = rules[column].map(_text)

    # Fare zones are those of the stops and those the fare rules refer to
    zones = [stops[["agency_id", "zone"]]] + [
        rules[["agency_id", c]].set_axis(["agency_id", "zone"], axis=1) for c in zone_columns
    ]
    zones = pd.concat(zones, ignore_index=True)
    zones = zones[zones.zone != ""].drop_duplicates()
    _check_ids(
        {
            "stop_id": (stops[["agency_id", "stop"]], True),
            "route_id": (patterns[["agency_id", "route"]].drop_duplicates(), True),
            "trip_id": (trips[["agency_id", "trip"]], True),
            "fare_id": (fares[["agency_id", "fare"]], True),
            "zone_id": (zones, False),
        },
        prefix_ids,
    )

    stop_times, trips = _stop_times(conn, ids, trips, stops.stop_id)
    stop_ids = pd.Series(_prefixed(prefix, stops, "stop").to_numpy(), index=stops.stop_id.to_numpy())
    trip_ids = pd.Series(_prefixed(prefix, trips, "trip").to_numpy(), index=trips.trip_id.to_numpy())

    agency_txt = pd.DataFrame(
        {
            "agency_id": agency.agency,
            "agency_name": agency.agency,
            "agency_url": agency.agency_url,
            "agency_timezone": agency.agency_timezone,
        }
    )
    services = _services(agency, trips.agency_id.unique())
    shapes = _shapes(conn, ids, trips.pattern_id.unique())

    tables = {
        "agency.txt": agency_txt if multiple else agency_txt.drop(columns=["agency_id"]),
        "stops.txt": _stops_txt(stops, prefix, set(stop_times.stop_id.unique())),
        "routes.txt": _routes_txt(patterns, prefix, names, multiple),
        "trips.txt": _trips_txt(trips, prefix, services, set(shapes.shape_id.unique())),
        "stop_times.txt": pd.DataFrame(
            {
                "trip_id": stop_times.trip_id.map(trip_ids).to_numpy(),
                "arrival_time": _gtfs_times(stop_times.arrival),
                "departure_time": _gtfs_times(stop_times.departure),
                "stop_id": stop_times.stop_id.map(stop_ids).to_numpy(),
                "stop_sequence": stop_times.seq.to_numpy() + 1,
            }
        ),
        "calendar_dates.txt": pd.DataFrame(
            {"service_id": services.service_id, "date": services["date"], "exception_type": 1}
        ).reset_index(drop=True),
    }
    if shapes.shape[0]:
        tables["shapes.txt"] = shapes
    fare_attributes, fare_rules = _fares(fares, rules, patterns, stops, prefix, names, multiple)
    if fare_attributes.shape[0]:
        tables["fare_attributes.txt"] = fare_attributes
    if fare_rules.shape[0]:
        tables["fare_rules.txt"] = fare_rules
    return tables


def write_gtfs(tables: dict[str, pd.DataFrame], file_path: os.PathLike | str) -> Path:
    """Writes GTFS files to a zip file, replacing any existing file only once all of them are written.

    :Arguments:
        **tables** (:obj:`dict`): Contents of each GTFS file, keyed by file name (e.g. 'stops.txt')

        **file_path** (:obj:`str` or :obj:`PathLike`): Path of the zip file to write

    :Returns:
        **file_path** (:obj:`Path`): Path of the zip file written
    """
    path = Path(file_path)
    partial = path.with_name(f"{path.name}.partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as feed:
            for name, table in tables.items():
                with feed.open(name, "w", force_zip64=table.shape[0] > ZIP64_ROWS) as raw:
                    with TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                        table.to_csv(text, index=False, lineterminator="\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _read(conn: Connection, sql: str, agency_ids: list, objects: bool = False) -> pd.DataFrame:
    """Reads a query for the agencies exported. With *objects*, values are kept as read (e.g. ``None`` for NULL)"""
    sql = sql.format(ids=",".join("?" * len(agency_ids)))
    if not objects:
        return pd.read_sql_query(sql, conn, params=agency_ids)
    cursor = conn.execute(sql, agency_ids)
    return pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description], dtype=object)


def _agencies(conn: Connection, agencies, agency_url, agency_timezone) -> pd.DataFrame:
    """Agencies to export, each with its URL and time zone"""
    table = pd.read_sql_query("SELECT agency_id, agency, service_date FROM agencies ORDER BY agency_id", conn)
    if table.empty:
        raise ValueError("The transit database has no agencies to export")

    available = ", ".join(f"{name!r} (agency_id {i})" for i, name in zip(table.agency_id, table.agency, strict=True))
    if agencies is not None:
        # Agencies are chosen by name (str) or by agency_id (int)
        wanted = [agencies] if isinstance(agencies, (str, Integral)) else list(agencies)
        by_id = table.agency_id.isin([int(a) for a in wanted if isinstance(a, Integral) and not isinstance(a, bool)])
        by_name = table.agency.isin([a for a in wanted if isinstance(a, str)])
        found = set(table.agency_id[by_id].tolist()) | set(table.agency[by_name].tolist())
        unknown = [a for a in wanted if isinstance(a, bool) or a not in found]
        if unknown:
            raise ValueError(f"Agencies {unknown} are not in the transit database, which has {available}")
        table = table[by_id | by_name].reset_index(drop=True)

    repeated = sorted(set(table.agency[table.agency.duplicated()]))
    if repeated:
        raise ValueError(
            f"More than one of the agencies to export is named {repeated}, and the agency name is the agency_id "
            f"written to the feed. Export them one at a time, choosing them by their agency_id ({available})"
        )

    def per_agency(value, argument: str, agency_id: int, name: str):
        if not isinstance(value, dict):
            return value
        for key in (name, agency_id):
            if key in value:
                return value[key]
        raise ValueError(f"{argument} has no entry for agency {name!r} (agency_id {agency_id})")

    table["agency_url"] = [
        per_agency(agency_url, "agency_url", i, name) for i, name in zip(table.agency_id, table.agency, strict=True)
    ]
    bad_urls = [url for url in table.agency_url if not str(url).lower().startswith(("http://", "https://"))]
    if bad_urls:
        raise ValueError(f"agency_url must be a full URL, starting with http:// or https://, not {bad_urls}")

    # Without the tz database installed, any time zone name is accepted
    time_zones = available_timezones()
    if (
        not isinstance(agency_timezone, str)
        or not agency_timezone
        or (time_zones and agency_timezone not in time_zones)
    ):
        msg = f"agency_timezone must be a tz database time zone (e.g. 'America/Santiago'), not {agency_timezone!r}"
        raise ValueError(msg)
    table["agency_timezone"] = agency_timezone
    return table


def _trips(conn: Connection, ids: list, patterns: pd.DataFrame) -> pd.DataFrame:
    """Trips of the agencies exported, with their GTFS route"""
    orphans = conn.execute("SELECT COUNT(*) FROM trips WHERE pattern_id NOT IN (SELECT pattern_id FROM routes)")
    orphans = orphans.fetchone()[0]
    if orphans:
        logger.warning(f"{orphans} trips belong to patterns absent from the routes table, so they are not exported")

    trips = _read(conn, TRIPS_SQL, ids).merge(patterns[["pattern_id", "agency_id", "route"]], on="pattern_id")
    no_id = trips.trip.isna() | (trips.trip.astype(str) == "")
    if no_id.any():
        logger.warning(f"{no_id.sum()} trips have no GTFS trip_id stored, so they are exported with their internal ID")
        trips.loc[no_id, "trip"] = trips.trip_id[no_id].astype(str)
    return trips


def _check_ids(ids: dict[str, tuple[pd.DataFrame, bool]], prefix_ids: bool) -> None:
    """Fails if GTFS IDs repeat within an agency or, unless they are to be prefixed, across agencies.

    *ids* holds, for each kind of ID, the ``agency_id`` and ID of each record and whether a repeated ID within an
    agency is an error"""
    within, across = [], []
    for kind, (records, unique) in ids.items():
        records = records.set_axis(["agency_id", "id"], axis=1)
        repeated = records.id[records.duplicated()].unique()
        if unique and repeated.size:
            within.append(f"{kind} {_examples(repeated)}")
        agencies_per_id = records.drop_duplicates().groupby("id").agency_id.nunique()
        shared = agencies_per_id.index[agencies_per_id > 1].to_numpy()
        if not prefix_ids and shared.size:
            across.append(f"{kind} {_examples(shared)}")

    if within:
        raise ValueError(f"These GTFS IDs repeat within an agency, which GTFS does not allow: {'; '.join(within)}")
    if across:
        raise ValueError(
            "The agencies to export share GTFS IDs, which would merge their records in a single feed: "
            f"{'; '.join(across)}. Use prefix_ids=True to prefix the IDs with the name of their agency, "
            "or export the agencies one at a time"
        )


def _stop_times(conn: Connection, ids: list, trips: pd.DataFrame, stop_ids: pd.Series) -> tuple:
    """Stop events of the trips whose schedule has one event per stop of their pattern, and those trips"""
    links = _read(conn, LINKS_SQL, ids)
    unlinked = np.setdiff1d(trips.pattern_id.unique(), links.pattern_id.unique())
    if unlinked.size:
        logger.warning(f"Patterns {_examples(unlinked)} have no route links, so their trips are not exported")
    sequences = _stop_sequences(links, stop_ids)
    schedule = _read(conn, SCHEDULE_SQL, ids).merge(trips[["trip_id", "pattern_id"]], on="trip_id")

    if (schedule[["arrival", "departure"]] < 0).any(axis=None):
        bad = schedule.trip_id[(schedule[["arrival", "departure"]] < 0).any(axis=1)].unique()
        raise ValueError(
            f"Trips {_examples(bad)} have times before midnight of the service day, which GTFS cannot hold"
        )

    # Each trip must have one event per stop of its pattern, numbered from zero
    by_trip = schedule.groupby("trip_id")
    numbered = (schedule.seq == by_trip.cumcount()).groupby(schedule.trip_id).all()
    stops_per_pattern = sequences.groupby("pattern_id").size()
    expected = pd.Series(trips.pattern_id.map(stops_per_pattern).to_numpy(), index=trips.trip_id.to_numpy())
    events = by_trip.size().reindex(expected.index)
    exported = (events == expected) & numbered.reindex(expected.index, fill_value=False)

    mismatched = expected.index[~exported & expected.notna()]
    if mismatched.size:
        gtfs_ids = trips.set_index("trip_id").trip.loc[mismatched].to_numpy()
        logger.warning(
            f"Trips {_examples(gtfs_ids)} do not have one scheduled time per stop of their pattern, "
            "so they are not exported"
        )
    schedule = schedule[schedule.trip_id.isin(expected.index[exported])]
    stop_times = schedule.merge(sequences, on=["pattern_id", "seq"], how="left")
    return stop_times, trips[trips.trip_id.isin(expected.index[exported])]


def _stop_sequences(links: pd.DataFrame, stop_ids: pd.Series) -> pd.DataFrame:
    """Stops of each pattern in sequence: stop k leaves on the k-th route link and the last stop ends the last one.

    Patterns whose route links skip a position, do not join up or use stops not exported are left out
    """
    by_pattern = links.groupby("pattern_id")
    position = by_pattern.cumcount()
    next_stop = by_pattern.from_stop.shift(-1)
    broken = links.seq - by_pattern.seq.transform("first") != position
    broken |= next_stop.notna() & (next_stop != links.to_stop)
    broken |= ~(links.from_stop.isin(stop_ids) & links.to_stop.isin(stop_ids))
    bad = links.pattern_id[broken].unique()
    if bad.size:
        logger.warning(
            f"The route links of patterns {_examples(bad)} do not join up as a sequence of stops exported, "
            "so their trips are not exported"
        )

    links = links.assign(position=position)[~links.pattern_id.isin(bad)]
    last = links.groupby("pattern_id").tail(1)
    return pd.concat(
        [
            pd.DataFrame({"pattern_id": links.pattern_id, "seq": links.position, "stop_id": links.from_stop}),
            pd.DataFrame({"pattern_id": last.pattern_id, "seq": last.position + 1, "stop_id": last.to_stop}),
        ],
        ignore_index=True,
    )


def _services(agency: pd.DataFrame, active_agencies) -> pd.DataFrame:
    """One service per agency with trips, running on the date imported, indexed by agency_id"""
    services = agency[agency.agency_id.isin(active_agencies)].set_index("agency_id")
    dates = {}
    for agency_id, row in services.iterrows():
        try:
            dates[agency_id] = date.fromisoformat(str(row.service_date)).strftime("%Y%m%d")
        except ValueError as e:
            raise ValueError(
                f"Agency {row.agency!r} has no valid service date stored ({row.service_date!r}), "
                "so the day its trips run on is unknown"
            ) from e
    services = services.assign(date=pd.Series(dates, dtype=str))
    return services.assign(service_id=services.agency.astype(str) + "_" + services["date"])


def _stops_txt(stops: pd.DataFrame, prefix: pd.Series, served: set) -> pd.DataFrame:
    """stops.txt, with the parent stations that are exported and can be stations"""
    # A parent station is a stop of the same agency, which is a station (location_type 1) only if no trip stops
    # there and it has no parent itself
    parents = stops[["agency_id", "stop", "stop_id", "parent_station"]].set_axis(
        ["agency_id", "parent_station", "parent_id", "grandparent"], axis=1
    )
    parent = stops[["agency_id", "parent_station"]].merge(parents, on=["agency_id", "parent_station"], how="left")
    has_parent = (stops.parent_station != "").to_numpy()
    keep = has_parent & parent.parent_id.notna().to_numpy()
    keep &= ~parent.parent_id.isin(served).to_numpy() & (parent.grandparent.fillna("") == "").to_numpy()
    if (has_parent & ~keep).any():
        logger.warning(
            f"{(has_parent & ~keep).sum()} stops are exported without their parent_station, as the parent is not "
            "among the stops exported (import keeps only stops served by trips) or cannot be a station"
        )
    stations = stops.stop_id.isin(parent.parent_id[keep])

    txt = pd.DataFrame(
        {
            "stop_id": _prefixed(prefix, stops, "stop"),
            "stop_name": stops.name,
            "stop_desc": stops.description,
            "stop_lat": stops.lat,
            "stop_lon": stops.lon,
            "zone_id": np.where(stops.zone != "", _prefixed(prefix, stops, "zone"), ""),
            "location_type": stations.astype(int),
            "parent_station": np.where(keep, _prefixed(prefix, stops, "parent_station"), ""),
        }
    )
    if (txt.stop_name == "").any():
        logger.warning(f"{(txt.stop_name == '').sum()} stops have no name, which GTFS requires")
    empty = [c for c in ["stop_desc", "zone_id", "parent_station"] if (txt[c] == "").all()]
    return txt.drop(columns=empty + ([] if stations.any() else ["location_type"]))


def _routes_txt(patterns: pd.DataFrame, prefix: pd.Series, names: pd.Series, multiple: bool) -> pd.DataFrame:
    """routes.txt, with each route described as its first pattern"""
    attributes = ["shortname", "longname", "description", "route_type"]
    differ = patterns.groupby(["agency_id", "route"])[attributes].nunique(dropna=False).gt(1).any(axis=1)
    if differ.any():
        logger.warning(
            f"The patterns of routes {_examples(differ.index[differ].get_level_values('route').unique())} differ "
            "in names, description or route type, so each route is exported as its first pattern"
        )

    routes = patterns.drop_duplicates(["agency_id", "route"])
    unnamed = (routes.shortname == "") & (routes.longname == "")
    if unnamed.any():
        logger.warning(f"Routes {_examples(routes.route[unnamed].unique())} have no name, which GTFS requires")
    txt = pd.DataFrame(
        {
            "route_id": _prefixed(prefix, routes, "route"),
            "agency_id": routes.agency_id.map(names),
            "route_short_name": routes.shortname,
            "route_long_name": routes.longname,
            "route_desc": routes.description,
            "route_type": routes.route_type,
        }
    )
    empty = ["route_desc"] if (txt.route_desc == "").all() else []
    return txt.drop(columns=empty + ([] if multiple else ["agency_id"])).reset_index(drop=True)


def _trips_txt(trips: pd.DataFrame, prefix: pd.Series, services: pd.DataFrame, shape_ids: set) -> pd.DataFrame:
    """trips.txt, with each trip's service and the shape of its pattern"""
    shape_id = trips.pattern_id.astype(str)
    txt = pd.DataFrame(
        {
            "route_id": _prefixed(prefix, trips, "route"),
            "service_id": trips.agency_id.map(services.service_id),
            "trip_id": _prefixed(prefix, trips, "trip"),
            "direction_id": trips.dir,
            "shape_id": shape_id.where(shape_id.isin(shape_ids), ""),
        }
    )
    return txt.drop(columns=["shape_id"] if (txt.shape_id == "").all() else []).reset_index(drop=True)


def _shapes(conn: Connection, ids: list, patterns: np.ndarray) -> pd.DataFrame:
    """shapes.txt, with one shape per pattern from its geometry or, for patterns without one, its route links.

    The geometry of a pattern is the best shape found on import: map-matched onto the network, or else the GTFS
    shape, or else straight lines between stops. Its route links cut that shape between stops, but, if
    map-matching was done after the links were built, they follow the GTFS shape instead."""
    points = _points(_read(conn, PATTERN_GEO_SQL, ids), patterns)
    missing = np.setdiff1d(patterns, points.pattern_id.unique())
    if missing.size:
        points = pd.concat([points, _points(_read(conn, LINK_GEO_SQL, ids), missing)], ignore_index=True)

    # Route links, and parts of a geometry, start where the previous one ends
    repeated = points[["pattern_id", "lat", "lon"]].eq(points[["pattern_id", "lat", "lon"]].shift()).all(axis=1)
    points = points[~repeated]
    points = points[points.groupby("pattern_id").pattern_id.transform("size") > 1]
    no_shape = np.setdiff1d(patterns, points.pattern_id.unique())
    if no_shape.size:
        logger.warning(f"Patterns {_examples(no_shape)} have no geometry, so their trips are exported without shape")

    return pd.DataFrame(
        {
            "shape_id": points.pattern_id.astype(str),
            "shape_pt_lat": points.lat,
            "shape_pt_lon": points.lon,
            "shape_pt_sequence": points.groupby("pattern_id").cumcount() + 1,
        }
    ).reset_index(drop=True)


def _points(geometries: pd.DataFrame, patterns: np.ndarray) -> pd.DataFrame:
    """Coordinates of the geometries of each pattern, in their order"""
    geometries = geometries[geometries.pattern_id.isin(patterns)]
    coords, index = shapely.get_coordinates(shapely.from_wkb(geometries.wkb.to_numpy()), return_index=True)
    return pd.DataFrame(
        {"pattern_id": geometries.pattern_id.to_numpy()[index], "lat": coords[:, 1], "lon": coords[:, 0]}
    )


def _fares(fares, rules, patterns, stops, prefix: pd.Series, names: pd.Series, multiple: bool) -> tuple:
    """fare_attributes.txt and fare_rules.txt, leaving out rules that refer to routes or stop zones not exported"""
    fare_ids = pd.Series(_prefixed(prefix, fares, "fare").to_numpy(), index=fares.fare_id.to_numpy())
    attributes = pd.DataFrame(
        {
            "fare_id": fare_ids.to_numpy(),
            "price": fares.price.map(_text),
            "currency_type": fares.currency.map(_text),
            "payment_method": fares.payment_method.map(_text),
            "transfers": fares.transfer.map(_text),
            "agency_id": fares.agency_id.map(names),
            "transfer_duration": fares.transfer_duration.map(_text),
        }
    )
    if not multiple:
        attributes = attributes.drop(columns=["agency_id"])

    # A rule for a route or zone not exported never applies, and without its route it would apply to every route
    routes = patterns.drop_duplicates("route_id")
    route = rules.route_id.map(pd.Series(_prefixed(prefix, routes, "route").to_numpy(), index=routes.route_id))
    zone_ids = set(zip(stops.agency_id, stops.zone, strict=True))
    unknown = rules.route_id.notna() & route.isna()
    for column in ["origin", "destination", "contains"]:
        known = [(a, z) in zone_ids for a, z in zip(rules.agency_id, rules[column], strict=True)]
        unknown |= (rules[column] != "") & ~np.array(known, dtype=bool)
    if unknown.any():
        logger.warning(f"{unknown.sum()} fare rules refer to routes or fare zones not exported, so they are left out")

    rules = rules[~unknown]
    zone = {c: np.where(rules[c] != "", _prefixed(prefix, rules, c), "") for c in ["origin", "destination", "contains"]}
    rules_txt = pd.DataFrame(
        {
            "fare_id": rules.fare_id.map(fare_ids),
            "route_id": route[~unknown].fillna(""),
            "origin_id": zone["origin"],
            "destination_id": zone["destination"],
            "contains_id": zone["contains"],
        }
    )
    return attributes.reset_index(drop=True), rules_txt.reset_index(drop=True)


def _prefixed(prefix: pd.Series, records: pd.DataFrame, column: str) -> pd.Series:
    """IDs in *column* of *records* as text, each with the prefix of its agency (empty unless IDs are prefixed)"""
    return records.agency_id.map(prefix).astype(str) + records[column].astype(str)


def _gtfs_times(seconds: pd.Series) -> np.ndarray:
    """Formats seconds after midnight as GTFS times (HH:MM:SS), with hours past 24 after midnight"""
    unique, inverse = np.unique(seconds.to_numpy(dtype=np.int64), return_inverse=True)
    text = np.array([f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" for s in unique.tolist()], dtype=object)
    return text[inverse.reshape(-1)]


def _text(value) -> str:
    """Formats a stored value as a GTFS field, with NULL as an empty field and whole numbers without decimals"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _examples(values, limit: int = 5) -> str:
    """Lists a few of the values given"""
    values = [str(v) for v in values]
    more = f" and {len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(repr(v) for v in sorted(values)[:limit]) + more
