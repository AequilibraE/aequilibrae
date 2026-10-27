import zipfile
from sqlite3 import Connection

import pandas as pd
import shapely

SERVICE_ID = "a.agency || '_' || replace(a.service_date, '-', '')"


def _hms(column: str) -> str:
    return f"printf('%02d:%02d:%02d', {column} / 3600, {column} % 3600 / 60, {column} % 60)"


def _id(column: str) -> str:
    # GTFS IDs are prefixed with the label of their agency on request, e.g. 'Metro:1', and also with the service
    # date of the import for agencies imported for several dates, e.g. 'Metro_20260105:1'
    repeated = "(SELECT COUNT(*) FROM agencies x WHERE x.agency = a.agency) > 1"
    return f"iif(:prefix, iif({repeated}, {SERVICE_ID}, a.agency) || ':', '') || {column}"


GTFS_SQL = {
    "agency.txt": """SELECT DISTINCT agency agency_id, agency agency_name, :url agency_url, :tz agency_timezone
                     FROM agencies""",
    # Stations are the stops that other stops have as their parent_station
    "stops.txt": f"""SELECT {_id("s.stop")} stop_id, s.name stop_name, s.description stop_desc,
                            ST_Y(s.geometry) stop_lat, ST_X(s.geometry) stop_lon,
                            {_id("NULLIF(s.transit_fare_zone, '')")} zone_id,
                            st.station IS NOT NULL location_type, {_id("p.stop")} parent_station
                     FROM stops s JOIN agencies a USING (agency_id)
                     LEFT JOIN stops p ON p.agency_id = s.agency_id AND p.stop = s.parent_station
                     LEFT JOIN (SELECT DISTINCT agency_id, parent_station station FROM stops) st
                          ON st.agency_id = s.agency_id AND st.station = s.stop""",
    "routes.txt": f"""SELECT {_id("r.route")} route_id, a.agency agency_id, r.shortname route_short_name,
                             r.longname route_long_name, r.description route_desc, r.route_type
                      FROM routes r JOIN agencies a USING (agency_id) GROUP BY r.agency_id, r.route""",
    "trips.txt": f"""SELECT {_id("r.route")} route_id, {SERVICE_ID} service_id, {_id("t.trip")} trip_id,
                            t.dir direction_id, t.pattern_id shape_id
                     FROM trips t JOIN routes r USING (pattern_id) JOIN agencies a USING (agency_id)""",
    # Stop k of a pattern starts its k-th route link, and the last stop ends its last link
    "stop_times.txt": f"""WITH pattern_stops AS (
                              SELECT pattern_id, seq, from_stop stop_id FROM route_links
                              UNION ALL
                              SELECT pattern_id, MAX(seq) + 1, to_stop FROM route_links GROUP BY pattern_id)
                          SELECT {_id("t.trip")} trip_id, {_hms("s.arrival")} arrival_time,
                                 {_hms("s.departure")} departure_time, {_id("st.stop")} stop_id,
                                 s.seq + 1 stop_sequence
                          FROM trips_schedule s JOIN trips t USING (trip_id)
                          JOIN routes r ON r.pattern_id = t.pattern_id JOIN agencies a ON a.agency_id = r.agency_id
                          JOIN pattern_stops p ON p.pattern_id = t.pattern_id AND p.seq = s.seq
                          JOIN stops st ON st.stop_id = CAST(p.stop_id AS TEXT)
                          ORDER BY s.trip_id, s.seq""",
    "calendar_dates.txt": f"""SELECT {SERVICE_ID} service_id, replace(a.service_date, '-', '') date,
                                     1 exception_type FROM agencies a""",
    "fare_attributes.txt": f"""SELECT {_id("f.fare")} fare_id, f.price, f.currency currency_type,
                                      CAST(f.payment_method AS TEXT) payment_method, CAST(f.transfer AS TEXT) transfers,
                                      a.agency agency_id,
                                      CAST(CAST(f.transfer_duration AS INTEGER) AS TEXT) transfer_duration
                               FROM fare_attributes f JOIN agencies a USING (agency_id)""",
    "fare_rules.txt": f"""SELECT {_id("f.fare")} fare_id,
                                 {_id("(SELECT route FROM routes r WHERE r.route_id = fr.route_id)")} route_id,
                                 {_id("fr.origin")} origin_id, {_id("fr.destination")} destination_id,
                                 {_id('fr."contains"')} contains_id
                          FROM fare_rules fr JOIN fare_attributes f USING (fare_id)
                          JOIN agencies a ON a.agency_id = f.agency_id""",
}

# GTFS IDs that agencies sharing them would repeat in a feed
UNIQUE_IDS = {
    "stops.txt": "stop_id",
    "routes.txt": "route_id",
    "trips.txt": "trip_id",
    "fare_attributes.txt": "fare_id",
}


def export_gtfs(conn: Connection, file_path, agency_url: str, agency_timezone: str, prefix_ids=False) -> None:
    params = {"url": agency_url, "tz": agency_timezone, "prefix": prefix_ids}
    tables = {name: pd.read_sql_query(sql, conn, params=params) for name, sql in GTFS_SQL.items()}
    for name, column in UNIQUE_IDS.items():
        if tables[name][column].duplicated().any():
            fix = "Import each agency once per date" if prefix_ids else "Export them with prefix_ids=True"
            raise ValueError(f"Agencies share IDs in {name}. {fix}")

    # One shape per pattern, from its geometry
    geo = pd.read_sql_query("SELECT pattern_id, ST_AsBinary(geometry) wkb FROM routes", conn)
    coords, index = shapely.get_coordinates(shapely.from_wkb(geo.wkb.to_numpy()), return_index=True)
    shapes = pd.DataFrame({"shape_id": geo.pattern_id.to_numpy()[index], "shape_pt_lat": coords[:, 1]})
    shapes = shapes.assign(shape_pt_lon=coords[:, 0], shape_pt_sequence=shapes.groupby("shape_id").cumcount() + 1)
    tables["shapes.txt"] = shapes

    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as feed:
        for name, table in tables.items():
            if not table.empty:
                feed.writestr(name, table.to_csv(index=False, lineterminator="\n"))
