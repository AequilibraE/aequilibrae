import sqlite3

import pandas as pd
from aequilibrae.utils.get_table import get_table

# from polarislib.network.data import DataTableStorage


def read_stop_times(conn: sqlite3.Connection):
    tpm = get_table("Transit_Pattern_Links", conn)
    tts = get_table("Transit_Trips_Schedule", conn).reset_index()
    tl = get_table("Transit_Links", conn).reset_index()
    trps = pd.read_sql("select pattern_id, trip_id from Transit_Trips", conn)
    tl.drop(columns=["pattern_id", "length", "geo", "type"], inplace=True)

    trip_stops = tts.merge(trps, on="trip_id")
    links = tpm.merge(tl, on="transit_link")

    first_nodes = links[["pattern_id", "from_node", "index"]].rename(columns={"from_node": "stop_id"})
    last_nodes = links.sort_values("index", ascending=False).drop_duplicates(subset=["pattern_id"], keep="first")
    last_nodes = last_nodes[["pattern_id", "to_node", "index"]].rename(columns={"to_node": "stop_id"})
    last_nodes.loc[:, "index"] += 1

    links = pd.concat([first_nodes, last_nodes], ignore_index=True).set_index(["pattern_id", "index"])
    stop_times = trip_stops.set_index(["pattern_id", "index"]).join(links).reset_index()
    renames = {"index": "stop_sequence", "departure": "departure_time", "arrival": "arrival_time"}
    stop_times.rename(columns=renames, inplace=True)

    # Conversion must be convoluted to support
    def pad(k: pd.Series) -> pd.Series:
        return k.astype(str).str.pad(width=2, side="left", fillchar="0")

    for field in ["departure_time", "arrival_time"]:
        h = pad(stop_times[field] // 3600)
        s = stop_times[field] % 3600
        m = pad(s // 60)
        s = pad(s % 60)
        stop_times[field] = h + ":" + m + ":" + s

    stop_times.loc[:, "stop_sequence"] += 1

    stop_times.stop_id = stop_times.stop_id.astype(str)

    return stop_times.sort_values(["trip_id", "stop_sequence"], ascending=True)
