import csv
from os.path import join

import pandas as pd

def write_trips(trips: pd.DataFrame, folder_path: str, service_dates: pd.Series):
    headers = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    all_trips = trips.reindex(columns=headers).copy()

    all_trips.to_csv(join(folder_path, "trips.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)

    services = all_trips[["service_id"]].drop_duplicates()
    min_date = str(service_dates.min()).replace("-", "")
    max_date = str(service_dates.max()).replace("-", "")
    services = services.assign(start_date=min_date, end_date=max_date)
    services = services.assign(monday=1, tuesday=1, wednesday=1, thursday=1, friday=1, saturday=1, sunday=1)
    services.to_csv(join(folder_path, "calendar.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
