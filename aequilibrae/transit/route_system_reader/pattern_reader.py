import sqlite3

import shapely.wkb
import shapely.wkt
from shapely.ops import transform
from aequilibrae.utils.get_table import get_table

from aequilibrae.transit.transit_elements import Pattern


def read_patterns(conn: sqlite3.Connection, transformer):
    patterns = []
    data = get_table("transit_patterns", conn).reset_index()
    if not data.shape[0]:
        return
    data.geo = data.geo.apply(shapely.wkb.loads)

    data.drop(columns=["matching_quality"], inplace=True)
    data.rename(columns={"pattern": "pattern_hash", "geo": "shape"}, inplace=True)

    for idx, dt in data.iterrows():
        pat = Pattern(None, dt.route_id, None).from_row(dt)
        pat.shape_length = pat.best_shape().length

        if transformer:
            pat.shape = transform(transformer.transform, pat.shape)

        patterns.append(pat)
    return patterns
