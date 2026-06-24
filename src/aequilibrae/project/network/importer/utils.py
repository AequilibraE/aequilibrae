import math

import geopandas as gpd
import pandas as pd
from pyproj import Geod

NODE_ID_START = 100000

# Above this bounding-box span (in degrees, max of width/height) a single UTM
# zone introduces unacceptable scale distortion, so we switch to geodesic
# (ellipsoidal) length computation instead of projecting to one local UTM CRS.
_MAX_UTM_SPAN_DEGREES = 3.0


def compute_lengths(geoms) -> pd.Series:
    """Length in metres for a GeoSeries of EPSG:4326 LineStrings.

    For small extents the geometries are projected to the estimated local UTM
    zone (fast, accurate). For large extents (state/national/continental) a
    single UTM zone would badly distort distances near its edges, so an
    ellipsoidal geodesic length is computed instead.
    """
    if not isinstance(geoms, gpd.GeoSeries):
        geoms = gpd.GeoSeries(geoms, crs="EPSG:4326")
    if geoms.crs is None:
        geoms = geoms.set_crs("EPSG:4326")

    minx, miny, maxx, maxy = geoms.total_bounds
    span = max(float(maxx - minx), float(maxy - miny))

    if span <= _MAX_UTM_SPAN_DEGREES:
        utm = geoms.estimate_utm_crs()
        return geoms.to_crs(utm).length.astype(float)

    return _geodesic_lengths(geoms)


def _geodesic_lengths(geoms) -> pd.Series:
    geod = Geod(ellps="WGS84")
    return pd.Series([float(geod.geometry_length(g)) for g in geoms], index=geoms.index, dtype=float)


# Number of fractional sample positions used when comparing two geometries.
_ALIGNMENT_SAMPLES = 16


def bearing_degrees(start, end) -> float:
    return math.degrees(math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0])))


def angular_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def line_straightness(geom) -> float:
    """Chord/length ratio in [0, 1]; 1 means perfectly straight."""
    coords = geom.coords
    if len(coords) < 2:
        return 1.0
    chord = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    length = float(geom.length)
    if length <= 0.0:
        return 1.0
    return max(0.0, min(1.0, chord / length))


def aligned_along_geometry(geom_a, geom_b, samples: int = _ALIGNMENT_SAMPLES) -> bool:
    """Whether ``geom_b`` traces roughly the same directed path as ``geom_a``.

    Samples each line at matching fractional distances and measures how well
    ``geom_b`` follows ``geom_a`` versus its reverse: for every sample point of
    ``geom_a`` we accumulate its distance to the same-fraction point of
    ``geom_b`` (forward) and to the mirror-fraction point (reverse). The smaller
    total wins. This is robust for near-coincident candidate geometries of
    arbitrary curvature, including curves, loops and roundabout segments where a
    global start-to-end vector is meaningless.
    """
    fractions = [i / samples for i in range(samples + 1)]
    pts_a = [geom_a.interpolate(f, normalized=True) for f in fractions]
    pts_b = [geom_b.interpolate(f, normalized=True) for f in fractions]

    forward_err = 0.0
    reverse_err = 0.0
    for k in range(samples + 1):
        forward_err += pts_a[k].distance(pts_b[k])
        reverse_err += pts_a[k].distance(pts_b[samples - k])
    return forward_err <= reverse_err


def compute_node_modes(node_ids, links: pd.DataFrame, fallback: str = "") -> list:
    nodes_col = pd.concat([links["a_node"], links["b_node"]], ignore_index=True)
    modes_col = pd.concat([links["modes"], links["modes"]], ignore_index=True).map(set)
    per_node = (
        pd.DataFrame({"node": nodes_col, "modes": modes_col})
        .groupby("node")["modes"]
        .agg(lambda s: "".join(sorted(set().union(*s))))
        .to_dict()
    )
    return [per_node.get(int(nid), fallback) for nid in node_ids]
