import sqlite3

import pandas as pd
from aequilibrae.utils.get_table import get_table

# from polarislib.network.data import DataTableStorage


def read_stop_times(conn: sqlite3.Connection):
    tts = get_table("trips_schedule", conn)
    links = get_table("route_links", conn)
    trps = pd.read_sql("SELECT pattern_id, trip_id FROM trips", conn)
    links.drop(columns=["distance", "geometry"], inplace=True)

    trip_stops = tts.merge(trps, on="trip_id")

    first_nodes = links[["pattern_id", "from_stop", "seq"]].rename(columns={"from_stop": "stop_id"})
    last_nodes = links.sort_values("seq", ascending=False).drop_duplicates(subset=["pattern_id"], keep="first")
    last_nodes = last_nodes[["pattern_id", "to_stop", "seq"]].rename(columns={"to_stop": "stop_id"})
    last_nodes.loc[:, "seq"] += 1

    links = pd.concat([first_nodes, last_nodes], ignore_index=True).set_index(["pattern_id", "seq"])
    stop_times = trip_stops.set_index(["pattern_id", "seq"]).join(links).reset_index()
    renames = {"seq": "stop_sequence", "departure": "departure_time", "arrival": "arrival_time"}
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
