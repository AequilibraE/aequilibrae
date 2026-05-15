import csv
from os.path import join

import pandas as pd


def write_fares(fare_attributes: pd.DataFrame, fare_rules: pd.DataFrame, folder_path: str):
    fattr = fare_attributes.copy()
    fattr.rename(columns={"currency": "currency_type", "transfer": "transfers"}, inplace=True)
    fattr.loc[:, "transfer_duration"] = fattr.transfer_duration.astype(int)

    headers = ["fare_id", "price", "currency_type", "payment_method", "transfers", "agency_id", "transfer_duration"]

    fattr[headers].to_csv(join(folder_path, "fare_attributes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)

    frls = fare_rules.copy()
    frls.rename(columns={fld: f"{fld}_id" for fld in frls.columns if "id" not in fld}, inplace=True)
    frls = frls[["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]]

    for fld in ["origin_id", "destination_id"]:
        frls.loc[:, fld] = frls[fld].astype("Int64").astype("string")
    frls.to_csv(join(folder_path, "fare_rules.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
