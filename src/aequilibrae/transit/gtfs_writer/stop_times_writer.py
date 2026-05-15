import csv
from pathlib import Path

import pandas as pd


def write_stop_times(stop_times: pd.DataFrame, folder_path: Path):
    columns = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    stop_times[columns].to_csv(folder_path / "stop_times.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
