import csv
from os.path import join
from typing import List

import numpy as np
import pandas as pd
from shapely.geometry import Point

from aequilibrae.transit.transit_elements import Pattern


def write_shapes(patterns: List[Pattern], folder_path: str):
    data = []
    for pat in patterns:
        points = [Point(pt) for pt in pat.shape.coords]
        lons = [pt.x for pt in points]
        lats = [pt.y for pt in points]
        distances = [0] + [x.distance(y) for x, y in zip(points[:-1], points[1:])]
        dt = pd.DataFrame(
            {
                "shape_id": pat.pattern_id,
                "shape_pt_lat": lats,
                "shape_pt_lon": lons,
                "shape_pt_sequence": np.arange(len(points)),
                "shape_dist_traveled": distances,
            }
        )
        dt.loc[:, "shape_dist_traveled"] = dt.shape_dist_traveled.cumsum()
        dt.loc[:, "shape_dist_traveled"] *= (dt.shape_dist_traveled.max()) * pat.shape_length / pat.shape.length
        data.append(dt)

    pd.concat(data).to_csv(join(folder_path, "shapes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
