import sqlite3
from math import log10, floor

import pandas as pd

from polarislib.network.traffic import Intersection
from polarislib.network.data import DataTableStorage
from polarislib.network.starts_logging import logger


def fix_connections_table(conn_tbl: pd.DataFrame, map_matching: pd.DataFrame, conn: sqlite3.Connection):
    data_tables = DataTableStorage()
    max_link = conn.execute("select max(link) from link").fetchone()[0]

    def build_key(df, max_link):
        multiplier = floor(log10(max_link)) + 3
        multiplier = pow(10, multiplier)
        key = df.link * multiplier + df.dir * (multiplier / 10) + df.to_link * 10 + df.to_dir
        return key

    conn_tbl = conn_tbl.assign(datakey=build_key(conn_tbl, max_link))

    map_matching.sort_values(by=["pattern_id", "index"], inplace=True)
    map_matching = map_matching.assign(
        to_link=map_matching.link.shift(-1),
        to_dir=map_matching.dir.shift(-1),
        to_pattern_id=map_matching.pattern_id.shift(-1),
    )
    map_matching = map_matching[map_matching.pattern_id == map_matching.to_pattern_id]
    map_matching.to_link = map_matching.to_link.astype(int)
    map_matching.to_dir = map_matching.to_dir.astype(int)
    map_matching = map_matching[(map_matching.link != map_matching.to_link) | (map_matching.dir != map_matching.to_dir)]
    map_matching = map_matching.assign(datakey=build_key(map_matching, max_link))

    missing_db = map_matching.loc[~map_matching.datakey.isin(conn_tbl.datakey)]
    if not missing_db.shape[0]:
        return
    conn_tbl = conn_tbl.assign(nodekey=10 * conn_tbl.link + conn_tbl.dir)
    missing_db = missing_db.assign(nodekey=10 * missing_db.link + missing_db.dir)
    missing_db = missing_db.merge(conn_tbl[["nodekey", "node"]], on="nodekey")
    missing_db = missing_db.drop_duplicates(subset=["link", "dir", "to_link", "to_dir"], ignore_index=True)
    if missing_db.shape[0] == 0:
        return
    logger.info(f"Adding {missing_db.shape[0]:,} new connections")
    for node, df in missing_db.groupby("node"):
        intersec = Intersection(data_tables, conn)
        intersec.load(node)
        for _, rec in df.iterrows():
            intersec.add_movement(rec.link, rec.to_link, "required_by_pt_map_matching")
