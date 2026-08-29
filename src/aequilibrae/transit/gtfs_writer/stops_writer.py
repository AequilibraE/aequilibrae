import csv
from pathlib import Path

import pandas as pd


def write_stops(stops: pd.DataFrame, folder_path: Path):
    headers = ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "zone_id", "parent_station"]
    df = stops.reindex(columns=headers).copy()
    for fld in ["stop_id", "zone_id", "parent_station"]:
        df[fld] = df[fld].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")

    df.to_csv(folder_path / "stops.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
