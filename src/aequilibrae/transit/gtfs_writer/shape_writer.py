import csv
from os.path import join
from typing import List, Union

import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.transit.transit_elements import Pattern


def _shape_points(shape: Union[LineString, MultiLineString]):
    if isinstance(shape, MultiLineString):
        coords = []
        for geom in shape.geoms:
            coords.extend(list(geom.coords))
        return [Point(pt) for pt in coords]
    return [Point(pt) for pt in shape.coords]


def write_shapes(patterns: Union[List[Pattern], pd.DataFrame], folder_path: str):
    if isinstance(patterns, pd.DataFrame):
        pattern_data = patterns.reindex(columns=["shape_id", "shape", "shape_length"]).copy()
        pattern_rows = pattern_data.to_dict(orient="records")
    else:
        pattern_rows = [
            {"shape_id": pat.pattern_id, "shape": pat.shape, "shape_length": pat.shape_length}
            for pat in patterns
            if pat.shape is not None
        ]

    data = []
    for pat in pattern_rows:
        if pat["shape"] is None:
            continue
        points = _shape_points(pat["shape"])
        lons = [pt.x for pt in points]
        lats = [pt.y for pt in points]
        distances = [0] + [x.distance(y) for x, y in zip(points[:-1], points[1:], strict=True)]
        dt = pd.DataFrame(
            {
                "shape_id": pat["shape_id"],
                "shape_pt_lat": lats,
                "shape_pt_lon": lons,
                "shape_pt_sequence": range(len(points)),
                "shape_dist_traveled": distances,
            }
        )
        dt.loc[:, "shape_dist_traveled"] = dt.shape_dist_traveled.cumsum()
        if dt.shape_dist_traveled.max() and pat["shape"].length:
            dt.loc[:, "shape_dist_traveled"] *= dt.shape_dist_traveled.max() * pat["shape_length"] / pat["shape"].length
        data.append(dt)

    output = pd.concat(data, ignore_index=True) if data else pd.DataFrame(
        columns=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"]
    )
    output.to_csv(join(folder_path, "shapes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
