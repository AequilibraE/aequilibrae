import math

import pandas as pd
from pyproj import Geod

NODE_ID_START = 100000

_MAX_UTM_SPAN_DEGREES = 3.0


def compute_lengths(geoms) -> pd.Series:
    """Lengths in metres, using local UTM or geodesic distance for wide extents."""
    minx, miny, maxx, maxy = geoms.total_bounds
    span = max(float(maxx - minx), float(maxy - miny))

    if span <= _MAX_UTM_SPAN_DEGREES:
        utm = geoms.estimate_utm_crs()
        return geoms.to_crs(utm).length.astype(float)

    geod = Geod(ellps="WGS84")
    return pd.Series([float(geod.geometry_length(g)) for g in geoms], index=geoms.index, dtype=float)


def bearing_degrees(start, end) -> float:
    return math.degrees(math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0])))


def angular_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def line_straightness(geom) -> float:
    """Chord/length ratio in [0, 1]; 1 means perfectly straight."""
    length = geom.length
    if length <= 0.0:
        return 1.0
    coords = geom.coords
    return min(1.0, math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]) / length)


def aligned_along_geometry(geom_a, geom_b, samples: int = 16) -> bool:
    """Return whether sampled points align better forward than in reverse."""
    fractions = [i / samples for i in range(samples + 1)]
    pts_a = [geom_a.interpolate(f, normalized=True) for f in fractions]
    pts_b = [geom_b.interpolate(f, normalized=True) for f in fractions]
    forward_err = sum(a.distance(b) for a, b in zip(pts_a, pts_b, strict=True))
    reverse_err = sum(a.distance(b) for a, b in zip(pts_a, reversed(pts_b), strict=True))
    return forward_err <= reverse_err


def compute_node_modes(node_ids, links: pd.DataFrame, fallback: str = "c") -> list:
    nodes_col = pd.concat([links["a_node"], links["b_node"]], ignore_index=True)
    modes_col = pd.concat([links["modes"], links["modes"]], ignore_index=True).map(set)
    per_node = (
        pd.DataFrame({"node": nodes_col, "modes": modes_col})
        .groupby("node")["modes"]
        .agg(lambda s: "".join(sorted(set().union(*s))))
        .to_dict()
    )
    return [per_node.get(int(nid), fallback) for nid in node_ids]
