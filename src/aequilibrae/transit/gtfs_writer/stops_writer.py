import csv
from os.path import join
from typing import List, Union

import pandas as pd

from aequilibrae.transit.transit_elements import Stop


def write_stops(stops: Union[List[Stop], pd.DataFrame], folder_path: str):
    headers = ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "zone_id", "parent_station"]
    if isinstance(stops, pd.DataFrame):
        df = stops.reindex(columns=headers).copy()
    else:
        data = [
            [
                stp.stop_id,
                stp.stop,
                stp.stop_name,
                stp.stop_desc,
                stp.stop_lat,
                stp.stop_lon,
                stp.zone_id,
                stp.parent_station,
            ]
            for stp in stops
        ]
        df = pd.DataFrame(data, columns=headers)

    for fld in ["stop_id", "zone_id", "parent_station"]:
        df[fld] = df[fld].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")

    df.to_csv(join(folder_path, "stops.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
