import csv
from pathlib import Path

import pandas as pd


def write_fares(fare_attributes: pd.DataFrame, fare_rules: pd.DataFrame, folder_path: Path):
    fattr = fare_attributes.rename(columns={"currency": "currency_type", "transfer": "transfers"})
    fattr.loc[:, "transfer_duration"] = fattr.transfer_duration.astype(int)

    headers = ["fare_id", "price", "currency_type", "payment_method", "transfers", "agency_id", "transfer_duration"]

    fattr[headers].to_csv(folder_path / "fare_attributes.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)

    frls = fare_rules.rename(columns={fld: f"{fld}_id" for fld in fare_rules.columns if "id" not in fld})
    frls = frls[["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]]

    for fld in ["origin_id", "destination_id"]:
        frls.loc[:, fld] = frls[fld].astype("Int64").astype("string")
    frls.to_csv(folder_path / "fare_rules.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
