import sqlite3

import shapely.wkb

from polarislib.network.data import DataTableStorage
from polarislib.network.transit.transit_elements import Stop


def read_stops(conn: sqlite3.Connection, transformer):
    data = DataTableStorage().get_table("transit_stops", conn).reset_index()
    data = data[data.agency_id > 1]
    data.geo = data.geo.apply(shapely.wkb.loads)
    if transformer:
        lons, lats = transformer.transform(data.X.values, data.Y.values)
        data.loc[:, "X"] = lons[:]
        data.loc[:, "Y"] = lats[:]

    data.drop(columns=["moved_by_matching", "Z"], inplace=True)
    data.rename(
        columns={
            "description": "stop_desc",
            "name": "stop_name",
            "street": "stop_street",
            "transit_zone_id": "zone_id",
            "Y": "stop_lat",
            "X": "stop_lon",
        },
        inplace=True,
    )

    return [Stop(-1).from_row(dt) for _, dt in data.iterrows()]
