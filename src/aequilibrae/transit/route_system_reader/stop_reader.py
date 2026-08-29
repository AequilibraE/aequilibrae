import sqlite3

import pandas as pd


def read_stops(conn: sqlite3.Connection):
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

    data = data.rename(columns={"x": "stop_lon", "y": "stop_lat"})
    for column in ["zone_id", "parent_station"]:
        if column not in data.columns:
            data[column] = pd.NA

    headers = ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "zone_id", "parent_station"]
    return data[headers].copy()
