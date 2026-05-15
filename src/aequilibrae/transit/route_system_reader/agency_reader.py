import sqlite3

import pandas as pd

from aequilibrae.utils.get_table import get_table


def read_agencies(conn: sqlite3.Connection):
    data = get_table("agencies", conn)
    data = data.loc[data.agency_id > 1, ["agency_id", "agency"]].copy()
    data.rename(columns={"agency": "agency_name"}, inplace=True)
    data = data.assign(agency_url="https://www.aequilibrae.com/", agency_timezone=pd.NA)
    return data[["agency_id", "agency_name", "agency_url", "agency_timezone"]]
