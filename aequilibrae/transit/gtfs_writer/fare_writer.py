import csv
from os.path import join

# from polarislib.network.data import DataTableStorage
from aequilibrae.utils.get_table import get_table


def write_fares(folder_path: str, conn):
    fattr = get_table("fare_attributes", conn).reset_index()
    fattr.rename(columns={"currency": "currency_type", "transfer": "transfers"}, inplace=True)
    fattr.loc[:, "transfer_duration"] = fattr.transfer_duration.astype(int)

    headers = ["fare_id", "price", "currency_type", "payment_method", "transfers", "agency_id", "transfer_duration"]

    fattr[headers].to_csv(join(folder_path, "fare_attributes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)

    frls = get_table("fare_rules", conn).reset_index()
    frls.rename(columns={fld: f"{fld}_id" for fld in frls.columns if "id" not in fld}, inplace=True)
    frls = frls[["fare_id", "route_id", "origin_id", "destination_id", "contains_id"]]

    for fld in ["origin_id", "destination_id"]:
        frls.loc[:, fld] = frls[fld].astype("Int64").astype("string")
    frls.to_csv(join(folder_path, "fare_rules.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
