import csv
from pathlib import Path

import pandas as pd


def write_agencies(agencies: pd.DataFrame, folder_path: Path):
    headers = ["agency_id", "agency_name", "agency_url", "agency_timezone"]
    df = agencies.reindex(columns=headers).copy()
    df.to_csv(folder_path / "agency.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
