import sqlite3

import pandas as pd


def read_stops(conn: sqlite3.Connection, transformer):
    sql = """
        SELECT stop_id,
               stop AS stop_code,
               name AS stop_name,
               description AS stop_desc,
               transit_fare_zone AS zone_id,
               parent_station,
               street AS stop_street,
               ST_X(geometry) AS x,
               ST_Y(geometry) AS y
        FROM stops
    """
    data = pd.read_sql(sql, conn)
    if transformer and not data.empty:
        lons, lats = transformer.transform(data["x"].tolist(), data["y"].tolist())
        data.loc[:, "stop_lon"] = lons[:]
        data.loc[:, "stop_lat"] = lats[:]
    else:
        data.loc[:, "stop_lon"] = data["x"]
        data.loc[:, "stop_lat"] = data["y"]

    data = data.drop(columns=["x", "y"], errors="ignore")
    for column in ["zone_id", "parent_station"]:
        if column not in data.columns:
            data[column] = pd.NA

    headers = ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "zone_id", "parent_station"]
    return data[headers].copy()
