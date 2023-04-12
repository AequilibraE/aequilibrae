import csv
from os.path import join

import pandas as pd


def write_stop_times(stop_times: pd.DataFrame, folder_path: str):
    columns = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    stop_times[columns].to_csv(join(folder_path, "stop_times.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
