import csv
from pathlib import Path

import pandas as pd


def write_routes(routes: pd.DataFrame, folder_path: Path):
    headers = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"]
    df = routes.reindex(columns=headers)
    df.to_csv(folder_path / "routes.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
