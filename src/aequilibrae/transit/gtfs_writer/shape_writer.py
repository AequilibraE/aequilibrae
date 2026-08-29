import csv
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


def _shape_points(shape: LineString | MultiLineString):
    if isinstance(shape, LineString):
        return [Point(pt) for pt in shape.coords]
    coords = []
    for geom in shape.geoms:
        coords.extend(list(geom.coords))
    return [Point(pt) for pt in coords]


def write_shapes(patterns: pd.DataFrame, folder_path: Path):
    pattern_data = patterns.reindex(columns=["shape_id", "shape", "shape_length"]).copy()
    pattern_rows = pattern_data.to_dict(orient="records")

    data = []
    for pat in pattern_rows:
        if pat["shape"] is None:
            continue
        points = _shape_points(pat["shape"])
        lons = [pt.x for pt in points]
        lats = [pt.y for pt in points]
        distances = [0.0] + [x.distance(y) for x, y in zip(points[:-1], points[1:], strict=True)]
        dt = pd.DataFrame(
            {
                "shape_id": pat["shape_id"],
                "shape_pt_lat": lats,
                "shape_pt_lon": lons,
                "shape_pt_sequence": range(len(points)),
                "shape_dist_traveled": distances,
            }
        )
        dt["shape_dist_traveled"] = dt["shape_dist_traveled"].cumsum()
        if dt.shape_dist_traveled.max() and pat["shape"].length:
            dt["shape_dist_traveled"] = dt["shape_dist_traveled"] * (pat["shape_length"] / pat["shape"].length)
        data.append(dt)

    output = pd.concat(data, ignore_index=True) if data else pd.DataFrame(
        columns=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"]
    )
    output.to_csv(folder_path / "shapes.txt", quoting=csv.QUOTE_NONNUMERIC, index=False)
