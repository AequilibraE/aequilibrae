import zipfile
from sqlite3 import Connection

import pandas as pd
import shapely

SERVICE_ID = "a.agency || '_' || replace(a.service_date, '-', '')"


def _hms(column: str) -> str:
    return f"printf('%02d:%02d:%02d', {column} / 3600, {column} % 3600 / 60, {column} % 60)"


GTFS_SQL = {
    "agency.txt": "SELECT agency agency_id, agency agency_name, :url agency_url, :timezone agency_timezone FROM agencies",
    "stops.txt": """SELECT stop stop_id, name stop_name, description stop_desc, ST_Y(geometry) stop_lat,
                           ST_X(geometry) stop_lon, transit_fare_zone zone_id FROM stops""",
    "routes.txt": """SELECT r.route route_id, a.agency agency_id, r.shortname route_short_name,
                            r.longname route_long_name, r.description route_desc, r.route_type
                     FROM routes r JOIN agencies a USING (agency_id) GROUP BY r.agency_id, r.route""",
    "trips.txt": f"""SELECT r.route route_id, {SERVICE_ID} service_id, t.trip trip_id, t.dir direction_id,
                            t.pattern_id shape_id
                     FROM trips t JOIN routes r USING (pattern_id) JOIN agencies a USING (agency_id)""",
    # Stop k of a pattern starts its k-th route link, and the last stop ends its last link
    "stop_times.txt": f"""WITH pattern_stops AS (
                              SELECT pattern_id, seq, from_stop stop_id FROM route_links
                              UNION ALL
                              SELECT pattern_id, MAX(seq) + 1, to_stop FROM route_links GROUP BY pattern_id)
                          SELECT t.trip trip_id, {_hms("s.arrival")} arrival_time, {_hms("s.departure")} departure_time,
                                 st.stop stop_id, s.seq + 1 stop_sequence
                          FROM trips_schedule s JOIN trips t USING (trip_id)
                          JOIN pattern_stops p ON p.pattern_id = t.pattern_id AND p.seq = s.seq
                          JOIN stops st ON st.stop_id = CAST(p.stop_id AS TEXT)
                          ORDER BY s.trip_id, s.seq""",
    "calendar_dates.txt": f"""SELECT {SERVICE_ID} service_id, replace(a.service_date, '-', '') date,
                                     1 exception_type FROM agencies a""",
    "fare_attributes.txt": """SELECT f.fare fare_id, f.price, f.currency currency_type,
                                     CAST(f.payment_method AS TEXT) payment_method, CAST(f.transfer AS TEXT) transfers,
                                     a.agency agency_id,
                                     CAST(CAST(f.transfer_duration AS INTEGER) AS TEXT) transfer_duration
                              FROM fare_attributes f JOIN agencies a USING (agency_id)""",
    "fare_rules.txt": """SELECT f.fare fare_id, (SELECT route FROM routes r WHERE r.route_id = fr.route_id) route_id,
                                fr.origin origin_id, fr.destination destination_id, fr."contains" contains_id
                         FROM fare_rules fr JOIN fare_attributes f USING (fare_id)""",
}


def export_gtfs(conn: Connection, file_path, agency_url: str, agency_timezone: str) -> None:
    params = {"url": agency_url, "timezone": agency_timezone}
    tables = {name: pd.read_sql_query(sql, conn, params=params) for name, sql in GTFS_SQL.items()}

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
