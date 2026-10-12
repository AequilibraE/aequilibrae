import geopandas as gpd
import logging
import math
import numpy as np
import shapely
import warnings
from shapely.geometry import Point

from aequilibrae.project.network.importer.simplifiers.common import (
    PROVENANCE_OUT_COL,
    SOURCE_ID_COL,
    build_oriented_source_attr_map,
    build_provenance,
    build_source_attr_map,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.importer.utils import (
    NODE_ID_START,
    aligned_along_geometry,
    angular_difference_degrees,
    bearing_degrees,
    compute_lengths,
    compute_node_modes,
    line_straightness,
)
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)

_DUAL_CARRIAGEWAY_WARNING = (
    "neatnet simplification may collapse parallel one-way carriageways into a single coarse link. "
    "When that happens, direction, speed, and lane fields are reconstructed heuristically after simplification."
)
_BEARING_MAX_DIFF_DEGREES = 35.0
_STRAIGHTNESS_THRESHOLD = 0.97
_DEKINK_MAX_POINTS = 6
_DEKINK_MIN_TURN_DEGREES = 25.0
_DEKINK_MAX_ENDPOINT_LENGTH = 0.00045
_DEFAULT_CONSOLIDATE_TOLERANCE = 10.0
_BUFFER_DIST = 25.0  # metres – search radius for matching original edges


def run_neatnet_simplify(
    net: StagedNetwork,
    *,
    consolidate_tolerance: float | None = _DEFAULT_CONSOLIDATE_TOLERANCE,
    simplification_factor: float = 2.0,
    min_dangle_length: float = 20.0,
) -> StagedNetwork:
    neatnet = require("neatnet", feature="neatnet simplification")

    warnings.warn(_DUAL_CARRIAGEWAY_WARNING, UserWarning, stacklevel=2)

    if len(net.links) == 0:
        return net

    if consolidate_tolerance is None:
        consolidate_tolerance = _DEFAULT_CONSOLIDATE_TOLERANCE

    geom_only = gpd.GeoDataFrame(geometry=net.links.geometry.to_crs(net.links.geometry.estimate_utm_crs()))

    if shapely.polygonize(geom_only.geometry.values).is_empty:
        logger.warning(
            "neatnet needs enclosed street blocks to detect face artifacts, and this network has none "
            "(it is tree-like, e.g. a sparse trail or rural network). Returning it unsimplified."
        )
        return net

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neatnet")
        try:
            simplified = neatnet.neatify(
                geom_only,
                consolidation_tolerance=float(consolidate_tolerance),
                simplification_factor=float(simplification_factor),
                min_dangle_length=float(min_dangle_length),
            ).to_crs("EPSG:4326")
        except KeyError as exc:
            if "face_artifact_index" not in str(exc):
                raise
            logger.warning(
                "neatnet could not derive face artifacts for this network (%s); returning it unsimplified.", exc
            )
            return net

    return _gdf_to_staged(simplified, original_links=net.links, source_meta=net.source_meta)


def _gdf_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    original_links: gpd.GeoDataFrame,
    source_meta: dict,
) -> StagedNetwork:
    edges = edges_gdf.reset_index(drop=True)
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)

    _transfer_attributes(edges, original_links)
    edges["geometry"] = [_dekink_endpoints_local(geom) for geom in edges.geometry]

    node_lookup, edges["a_node"], edges["b_node"] = _build_endpoint_index(edges.geometry.values)
    edges["distance"] = compute_lengths(edges.geometry).to_numpy()

    nodes = gpd.GeoDataFrame(
        {"node_id": list(node_lookup.values()), "geometry": [Point(xy) for xy in node_lookup]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes["modes"] = compute_node_modes(nodes["node_id"].to_numpy(), edges)

    return StagedNetwork(nodes=nodes, links=edges, source_meta=source_meta)


def _build_endpoint_index(geoms):
    starts = shapely.get_coordinates(shapely.get_point(geoms, 0))
    ends = shapely.get_coordinates(shapely.get_point(geoms, -1))
    node_lookup = {}
    a_nodes, b_nodes = [], []
    for start, end in zip(starts, ends, strict=True):
        for xy, target in ((start, a_nodes), (end, b_nodes)):
            key = (round(float(xy[0]), 7), round(float(xy[1]), 7))
            target.append(node_lookup.setdefault(key, NODE_ID_START + len(node_lookup)))
    return node_lookup, np.array(a_nodes, dtype=np.int64), np.array(b_nodes, dtype=np.int64)


def _transfer_attributes(simplified: gpd.GeoDataFrame, original: gpd.GeoDataFrame) -> None:
    """Match each simplified edge to nearby originals and aggregate attributes."""
    utm = simplified.geometry.estimate_utm_crs()
    simp_geoms = simplified.geometry.to_crs(utm).values
    orig_geoms = original.geometry.to_crs(utm).values
    src_attrs = build_source_attr_map(original)
    oriented_src_attrs = build_oriented_source_attr_map(original)

    tree = shapely.STRtree(orig_geoms)

    attributes = []

    orig_dir = original["direction"].to_numpy()
    orig_modes = original["modes"].to_numpy()
    orig_lt = original["link_type"].to_numpy()
    orig_name = original["name"].to_numpy()
    orig_source_ids = original[SOURCE_ID_COL].astype(str).to_numpy()
    orig_straightness = np.array([line_straightness(g) for g in orig_geoms], dtype=float)

    for sg in simp_geoms:
        nearest_oidx = int(tree.nearest(sg))
        hits = tree.query(sg.buffer(_BUFFER_DIST))
        if len(hits) == 0:
            hits = [nearest_oidx]

        nearest_lt = str(orig_lt[nearest_oidx])
        compatible = [int(oidx) for oidx in hits if _link_type_compatible(nearest_lt, str(orig_lt[oidx]))]
        reduced = _reduce_candidates_by_overlap(sg, orig_geoms, compatible)
        fwd_candidates, bwd_candidates = _classify_candidates(
            sg, reduced, orig_geoms, orig_straightness, orig_dir, orig_source_ids
        )

        ordered_source_ids = _ordered_source_ids(fwd_candidates + bwd_candidates)
        sources = [oidx for oidx, _dist in reduced] or [nearest_oidx]
        all_modes = set().union(*(orig_modes[o] for o in sources if isinstance(orig_modes[o], str)))
        attributes.append(
            {
                "direction": int(bool(fwd_candidates)) - int(bool(bwd_candidates)),
                "modes": "".join(sorted(all_modes)) or "c",
                "link_type": nearest_lt,
                "name": orig_name[nearest_oidx],
                "speed_ab": _nearest_oriented_value(fwd_candidates, oriented_src_attrs, "speed"),
                "speed_ba": _nearest_oriented_value(bwd_candidates, oriented_src_attrs, "speed"),
                "lanes_ab": _nearest_oriented_value(fwd_candidates, oriented_src_attrs, "lanes"),
                "lanes_ba": _nearest_oriented_value(bwd_candidates, oriented_src_attrs, "lanes"),
                SOURCE_ID_COL: ordered_source_ids[0] if ordered_source_ids else orig_source_ids[nearest_oidx],
                PROVENANCE_OUT_COL: build_provenance(ordered_source_ids, src_attrs),
            }
        )

    for column in attributes[0]:
        simplified[column] = [attrs[column] for attrs in attributes]


_LINK_TYPE_FAMILIES = (
    {"motorway", "motorway_link", "trunk", "trunk_link"},
    {
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "road",
        "busway",
        "bus_guideway",
    },
    {"footway", "pedestrian", "steps", "path", "corridor", "elevator", "escalator", "bridleway"},
    {"cycleway"},
)


_FAMILY_OF = {lt: i for i, family in enumerate(_LINK_TYPE_FAMILIES) for lt in family}


def _link_type_compatible(reference: str, candidate: str) -> bool:
    """Allow attribute transfer within a road family or when either type is unknown."""
    ref_family = _FAMILY_OF.get((reference or "").lower())
    cand_family = _FAMILY_OF.get((candidate or "").lower())
    return ref_family is None or cand_family is None or ref_family == cand_family


def _geometry_summary(geom) -> dict:
    coords = geom.coords
    start = coords[0]
    end = coords[-1]
    return {
        "start": start,
        "end": end,
        "bearing": bearing_degrees(start, end),
        "straightness": line_straightness(geom),
    }


def _reduce_candidates_by_overlap(simplified_geom, orig_geoms, compatible: list[int]) -> list[tuple[int, float]]:
    # Cheap proxy ranking; exact line/buffer intersection lengths proved too slow.
    simp_buffer = simplified_geom.buffer(_BUFFER_DIST)
    sx0, sy0 = simplified_geom.coords[0]
    sx1, sy1 = simplified_geom.coords[-1]
    scored = []
    for oidx in compatible:
        og = orig_geoms[oidx]
        intersects = 1 if simp_buffer.intersects(og) else 0
        dist = float(simplified_geom.distance(og))
        (ox0, oy0), (ox1, oy1) = og.coords[0], og.coords[-1]
        endpoint_cost = min(
            math.hypot(sx0 - ox0, sy0 - oy0) + math.hypot(sx1 - ox1, sy1 - oy1),
            math.hypot(sx0 - ox1, sy0 - oy1) + math.hypot(sx1 - ox0, sy1 - oy0),
        )
        scored.append((oidx, intersects, dist, endpoint_cost))
    scored.sort(key=lambda item: (-item[1], item[2], item[3]))
    return [(oidx, dist) for oidx, _intersects, dist, _endpoint_cost in scored[:4]]


def _classify_candidates(simp_geom, reduced, orig_geoms, orig_straightness, orig_dir, orig_source_ids):
    simp_summary = _geometry_summary(simp_geom)
    fwd_candidates, bwd_candidates = [], []
    for oidx, dist in reduced:
        aligned = _classify_orientation_fast(simp_summary, orig_geoms[oidx], orig_straightness[oidx])
        if aligned is None:
            aligned = aligned_along_geometry(simp_geom, orig_geoms[oidx])
        along, against = (fwd_candidates, bwd_candidates) if aligned else (bwd_candidates, fwd_candidates)
        if orig_dir[oidx] != -1:
            along.append((f"{orig_source_ids[oidx]}::ab", dist))
        if orig_dir[oidx] != 1:
            against.append((f"{orig_source_ids[oidx]}::ba", dist))
    return fwd_candidates, bwd_candidates


def _classify_orientation_fast(simp_summary: dict, orig_geom, orig_straightness: float) -> bool | None:
    if simp_summary["straightness"] < _STRAIGHTNESS_THRESHOLD or orig_straightness < _STRAIGHTNESS_THRESHOLD:
        return None
    (sx0, sy0), (sx1, sy1) = simp_summary["start"], simp_summary["end"]
    orig_coords = orig_geom.coords
    (ox0, oy0), (ox1, oy1) = orig_coords[0], orig_coords[-1]
    orig_bearing = bearing_degrees((ox0, oy0), (ox1, oy1))

    same_cost = math.hypot(sx0 - ox0, sy0 - oy0) + math.hypot(sx1 - ox1, sy1 - oy1)
    rev_cost = math.hypot(sx0 - ox1, sy0 - oy1) + math.hypot(sx1 - ox0, sy1 - oy0)
    bearing_fwd = angular_difference_degrees(simp_summary["bearing"], orig_bearing)
    bearing_rev = angular_difference_degrees(simp_summary["bearing"], (orig_bearing + 180.0) % 360.0)

    if same_cost < rev_cost and bearing_fwd <= _BEARING_MAX_DIFF_DEGREES:
        return True
    if rev_cost < same_cost and bearing_rev <= _BEARING_MAX_DIFF_DEGREES:
        return False
    return None


def _dekink_endpoints_local(geom):
    coords = list(geom.coords)
    if len(coords) < 4:
        return geom

    coords = _prune_endpoint_kinks(coords, reverse=False)
    coords = _prune_endpoint_kinks(coords, reverse=True)
    return shapely.LineString(coords)


def _prune_endpoint_kinks(coords: list, *, reverse: bool) -> list:
    work = list(reversed(coords)) if reverse else list(coords)
    max_steps = min(_DEKINK_MAX_POINTS, len(work) - 3)

    for _ in range(max_steps):
        a, b, c = work[0], work[1], work[2]
        ang1 = bearing_degrees(a, b)
        ang2 = bearing_degrees(b, c)
        if angular_difference_degrees(ang1, ang2) < _DEKINK_MIN_TURN_DEGREES:
            break
        trial = [work[0]] + work[2:]
        if shapely.LineString(trial).is_simple:
            work = trial
        else:
            break

    for idx in range(2, min(len(work) - 1, _DEKINK_MAX_POINTS + 1)):
        chain = work[: idx + 1]
        chain_len = shapely.LineString(chain).length
        chord_len = shapely.LineString([chain[0], chain[-1]]).length
        if chord_len <= 0 or chain_len > _DEKINK_MAX_ENDPOINT_LENGTH:
            break
        if chain_len / chord_len < 1.02:
            continue
        stable_bearing = bearing_degrees(chain[-1], work[idx + 1])
        approach_bearing = bearing_degrees(chain[0], chain[-1])
        if angular_difference_degrees(approach_bearing, stable_bearing) > 55.0:
            continue
        trial = [work[0], work[idx]] + work[idx + 1 :]
        if shapely.LineString(trial).is_simple:
            work = trial
            break

    return list(reversed(work)) if reverse else work


def _nearest_oriented_value(candidates: list[tuple[str, float]], oriented_src_attrs: dict, field: str):
    for source_ref, _dist in sorted(candidates, key=lambda item: item[1]):
        value = oriented_src_attrs.get(source_ref, {}).get(field)
        if value is not None:
            return value
    return None


def _ordered_source_ids(candidates: list[tuple[str, float]]) -> list[str]:
    """Base source ids from ``(source_ref, distance)`` candidates, nearest first, de-duplicated."""
    source_ids = []
    for source_ref, _dist in sorted(candidates, key=lambda item: item[1]):
        source_id = source_ref.partition("::")[0]
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids
