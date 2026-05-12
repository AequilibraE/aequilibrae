import pandas as pd

NODE_ID_START = 100000

# Above this bounding-box span (in degrees, max of width/height) a single UTM
# zone introduces unacceptable scale distortion, so we switch to geodesic
# (ellipsoidal) length computation instead of projecting to one local UTM CRS.
_MAX_UTM_SPAN_DEGREES = 3.0


def compute_lengths(geoms) -> "pd.Series":
    """Length in metres for a GeoSeries of EPSG:4326 LineStrings.

    For small extents the geometries are projected to the estimated local UTM
    zone (fast, accurate). For large extents (state/national/continental) a
    single UTM zone would badly distort distances near its edges, so an
    ellipsoidal geodesic length is computed instead.
    """
    import geopandas as gpd

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


def _geodesic_lengths(geoms) -> "pd.Series":
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    values = [
        0.0 if (g is None or getattr(g, "is_empty", False)) else float(geod.geometry_length(g)) for g in geoms
    ]
    return pd.Series(values, index=geoms.index, dtype=float)

# Number of fractional sample positions used when comparing two geometries.
_ALIGNMENT_SAMPLES = 16


def aligned_along_geometry(geom_a, geom_b, samples: int = _ALIGNMENT_SAMPLES) -> bool:
    """Whether ``geom_b`` traces roughly the same directed path as ``geom_a``.

    The legacy implementation compared the *global* start->end vectors of the two
    lines. That is meaningless for curved roads, U-shaped ramps, loop ramps and
    roundabout segments, where the start->end chord says nothing about the
    direction of travel along the geometry (and is degenerate for closed loops).

    This implementation samples each line at matching fractional distances and
    measures how well ``geom_b`` follows ``geom_a`` versus its reverse: for every
    sample point of ``geom_a`` we accumulate its distance to the same-fraction
    point of ``geom_b`` (forward) and to the mirror-fraction point (reverse). The
    smaller total wins. This is robust for near-coincident candidate geometries
    of arbitrary curvature, which is exactly how the simplifiers use it (matching
    a simplified link against the original links that lie on top of it).
    """
    if geom_a is None or geom_b is None:
        return True
    if getattr(geom_a, "is_empty", False) or getattr(geom_b, "is_empty", False):
        return True

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
