import csv
from os.path import join

import pandas as pd

# from polarislib.network.data import DataTableStorage
from aequilibrae.utils.get_table import get_table


def write_fares(folder_path: str, conn):
    fattr = get_table("Transit_Fare_Attributes", conn).reset_index()
    fattr.rename(columns={"currency": "currency_type", "transfer": "transfers"}, inplace=True)
    fattr.transfer_duration = fattr.transfer_duration.astype(int)

    headers = ["fare_id", "price", "currency_type", "payment_method", "transfers", "agency_id", "transfer_duration"]

    fattr[headers].to_csv(join(folder_path, "fare_attributes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)

    frls = get_table("Transit_Fare_Rules", conn).reset_index()
    frls.rename(columns={fld: f"{fld}_id" for fld in frls.columns if "id" not in fld}, inplace=True)
    frls = frls[["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]]

    for fld in ["origin_id", "destination_id"]:
        frls[fld].fillna(-99999, inplace=True)
        frls[fld] = frls[fld].astype(int).astype(str)
        frls.loc[frls[fld] == "-99999", fld] = pd.NA
    frls.to_csv(join(folder_path, "fare_rules.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
