import sqlite3

import pandas as pd
import shapely.wkt
from shapely.ops import transform


def read_patterns(conn: sqlite3.Connection, transformer):
    sql = "SELECT pattern_id, ST_AsText(geometry) AS shape_wkt FROM routes WHERE geometry IS NOT NULL"
    data = pd.read_sql(sql, conn)
    if not data.shape[0]:
        return pd.DataFrame(columns=["shape_id", "shape", "shape_length"])
    data = data.loc[data.shape_wkt.notna(), ["pattern_id", "shape_wkt"]].copy()
    data.loc[:, "shape"] = data["shape_wkt"].apply(shapely.wkt.loads)
    if transformer:
        data.loc[:, "shape"] = data["shape"].apply(lambda geom: transform(transformer.transform, geom))
    data = data.rename(columns={"pattern_id": "shape_id"})
    data = data.assign(shape_length=data["shape"].apply(lambda geom: geom.length))
    return data[["shape_id", "shape", "shape_length"]].copy()
