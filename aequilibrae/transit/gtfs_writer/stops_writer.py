import csv
from os.path import join
from typing import List

import numpy as np
import pandas as pd

from aequilibrae.transit.transit_elements import Stop


def write_stops(stops: List[Stop], folder_path: str):
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

    headers = ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "zone_id", "parent_station"]
    df = pd.DataFrame(data, columns=headers)

    df.parent_station = df.parent_station.astype(float)
    df.loc[:, "parent_station"] = np.NAN
    for fld in ["zone_id", "stop_id"]:
        df[fld] = df[fld].astype(float)
        df[fld].fillna(-99999, inplace=True)
        df[fld] = df[fld].astype(int)
        df[fld] = df[fld].astype(str)
        df.loc[df[fld] == "-99999", fld] = ""

    df.to_csv(join(folder_path, "stops.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
