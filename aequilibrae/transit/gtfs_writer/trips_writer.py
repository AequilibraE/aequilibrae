import csv
import sqlite3
from os.path import join
from typing import List

import pandas as pd

from aequilibrae.transit.transit_elements import Trip


def write_trips(trips: List[Trip], folder_path: str, conn: sqlite3.Connection):
    headers = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    data = [[trp.__dict__[hdr] for hdr in headers] for trp in trips]

    all_trips = pd.DataFrame(data, columns=headers)
    all_trips.to_csv(join(folder_path, "trips.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)

    srvc = pd.read_sql("select service_date from Transit_Agencies where agency_id>1", conn)

    services = all_trips[["service_id"]].drop_duplicates()
    min_date = str(srvc.service_date.min()).replace("-", "")
    max_date = str(srvc.service_date.max()).replace("-", "")
    services = services.assign(start_date=min_date, end_date=max_date)
    services = services.assign(monday=1, tuesday=1, wednesday=1, thursday=1, friday=1, saturday=1, sunday=1)
    services.to_csv(join(folder_path, "calendar.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
