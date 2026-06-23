import geopandas as gpd
import logging
import math
import numpy as np
import shapely
import warnings
from shapely.geometry import Point

from aequilibrae.project.network.importer.simplifiers.impl_osmnx import (
    _build_oriented_source_attr_map,
    _build_provenance,
    _build_source_attr_map,
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
_PROVENANCE_OUT_COL = "source_ids"
_SOURCE_ID_COL = "source_id"
_SOURCE_REFS_TMP_COL = "_source_refs"
_BEARING_MAX_DIFF_DEGREES = 35.0
_STRAIGHTNESS_THRESHOLD = 0.97
_DEKINK_MAX_POINTS = 6
_DEKINK_MIN_TURN_DEGREES = 25.0
_DEKINK_MAX_ENDPOINT_LENGTH = 0.00045


def run_neatnet_simplify(net: StagedNetwork, *, exclusion_mask=None, metrics=None, **_) -> StagedNetwork:
    require("neatnet", feature="neatnet simplification")

    import neatnet

    warnings.warn(_DUAL_CARRIAGEWAY_WARNING, UserWarning, stacklevel=2)
    logger.warning(_DUAL_CARRIAGEWAY_WARNING)

    if len(net.links) == 0:
        return net

    edges = net.links.copy()
    utm = edges.geometry.estimate_utm_crs()
    geom_only = gpd.GeoDataFrame(geometry=edges.geometry, crs=edges.crs).to_crs(utm)

    neatify_kwargs = {}
    if exclusion_mask is not None:
        neatify_kwargs["exclusion_mask"] = exclusion_mask.to_crs(utm).geometry

    from time import perf_counter

    t0 = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neatnet")
        simplified = neatnet.neatify(geom_only, **neatify_kwargs).to_crs("EPSG:4326")
    if metrics is not None:
        metrics["neatify_time_s"] = perf_counter() - t0

    return _gdf_to_staged(simplified, original_links=edges, source_meta=net.source_meta, metrics=metrics)


def _gdf_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    original_links: gpd.GeoDataFrame,
    source_meta: dict,
    metrics: dict | None = None,
) -> StagedNetwork:
    edges = edges_gdf.copy().reset_index(drop=True)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    coords, indices = shapely.get_coordinates(edges.geometry, return_index=True)
    last_pos = np.searchsorted(indices, np.arange(len(edges)), side="right") - 1
    first_pos = np.searchsorted(indices, np.arange(len(edges)), side="left")
    starts = coords[first_pos]
    ends = coords[last_pos]

    endpoints = {}
    a_nodes = np.empty(len(edges), dtype=np.int64)
    b_nodes = np.empty(len(edges), dtype=np.int64)
    next_id = NODE_ID_START
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for arr, target in ((start, a_nodes), (end, b_nodes)):
            key = (round(float(arr[0]), 7), round(float(arr[1]), 7))
            nid = endpoints.get(key)
            if nid is None:
                nid = endpoints[key] = next_id
                next_id += 1
            target[i] = nid

    edges["a_node"] = a_nodes
    edges["b_node"] = b_nodes
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)

    _transfer_attributes(edges, original_links, metrics=metrics)
    edges["geometry"] = [_dekink_endpoints_local(geom) for geom in edges.geometry]

    edges["distance"] = compute_lengths(edges.geometry).to_numpy()

    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(endpoints.values()),
            "geometry": [Point(x, y) for x, y in endpoints],
            "modes": "c",
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes["modes"] = compute_node_modes(nodes["node_id"].to_numpy(), edges, fallback="c")

    return StagedNetwork(nodes=nodes, links=edges, source_meta=source_meta)


_BUFFER_DIST = 25.0  # metres – search radius for matching original edges


def _transfer_attributes(simplified: gpd.GeoDataFrame, original: gpd.GeoDataFrame, metrics: dict | None = None) -> None:
    """Match each simplified edge to nearby originals and aggregate attributes."""
    from time import perf_counter

    t0 = perf_counter()
    utm = simplified.geometry.estimate_utm_crs()
    simp_geoms = simplified.geometry.to_crs(utm).values
    orig_geoms = original.geometry.to_crs(utm).values
    src_attrs = _build_source_attr_map(original)
    oriented_src_attrs = _build_oriented_source_attr_map(original)

    tree = shapely.STRtree(orig_geoms)

    n = len(simplified)
    directions = np.zeros(n, dtype=int)
    modes_arr = ["c"] * n
    link_types = ["unknown"] * n
    names = [None] * n
    speed_ab = [None] * n
    speed_ba = [None] * n
    lanes_ab = [None] * n
    lanes_ba = [None] * n
    primary_source_ids = [None] * n
    provenance = [None] * n
    source_refs = [[] for _ in range(n)]

    orig_dir = original["direction"].to_numpy()
    orig_modes = original["modes"].to_numpy()
    orig_lt = original["link_type"].to_numpy()
    orig_name = original["name"].to_numpy()
    orig_source_ids = original[_SOURCE_ID_COL].astype(str).to_numpy()
    orig_straightness = np.array([line_straightness(g) for g in orig_geoms], dtype=float)

    raw_hit_counts = []
    reduced_counts = []
    bearing_counts = []
    fallback_counts = []

    for i in range(n):
        sg = simp_geoms[i]
        hits = tree.query(sg.buffer(_BUFFER_DIST))
        if len(hits) == 0:
            hits = np.array([tree.nearest(sg)])
        raw_hit_counts.append(int(len(hits)))

        nearest_oidx = int(tree.nearest(sg))
        nearest_lt = str(orig_lt[nearest_oidx])
        simp_summary = _geometry_summary(sg)

        compatible = [int(oidx) for oidx in hits if _link_type_compatible(nearest_lt, str(orig_lt[oidx]))]
        reduced = _reduce_candidates_by_overlap(sg, orig_geoms, compatible)
        reduced_counts.append(int(len(reduced)))

        (
            fwd_candidates,
            bwd_candidates,
            contributing_oidx,
            bearing_direct,
            fallback,
        ) = _classify_candidates(
            reduced=reduced,
            simp_geom=sg,
            simp_summary=simp_summary,
            orig_geoms=orig_geoms,
            orig_straightness=orig_straightness,
            orig_dir=orig_dir,
            orig_source_ids=orig_source_ids,
            metrics=metrics,
        )

        bearing_counts.append(bearing_direct)
        fallback_counts.append(fallback)

        has_fwd = len(fwd_candidates) > 0
        has_bwd = len(bwd_candidates) > 0

        if has_fwd and has_bwd:
            directions[i] = 0
        elif has_fwd:
            directions[i] = 1
        elif has_bwd:
            directions[i] = -1
        link_types[i] = nearest_lt
        names[i] = orig_name[nearest_oidx]

        ordered_refs = _unique_source_refs(fwd_candidates + bwd_candidates)
        source_refs[i] = ordered_refs
        ordered_source_ids = _ordered_source_ids(ordered_refs)
        primary_source_ids[i] = ordered_source_ids[0] if ordered_source_ids else orig_source_ids[nearest_oidx]
        provenance[i] = _build_provenance(ordered_source_ids, src_attrs)

        all_modes: set = set()
        for oidx in contributing_oidx:
            m = orig_modes[oidx]
            if isinstance(m, str):
                all_modes.update(m)
        if not all_modes:
            m = orig_modes[nearest_oidx]
            if isinstance(m, str):
                all_modes.update(m)
        modes_arr[i] = "".join(sorted(all_modes)) or "c"

        if fwd_candidates:
            speed_ab[i] = _nearest_oriented_value(fwd_candidates, oriented_src_attrs, "speed")
            lanes_ab[i] = _nearest_oriented_value(fwd_candidates, oriented_src_attrs, "lanes")

        if bwd_candidates:
            speed_ba[i] = _nearest_oriented_value(bwd_candidates, oriented_src_attrs, "speed")
            lanes_ba[i] = _nearest_oriented_value(bwd_candidates, oriented_src_attrs, "lanes")

    simplified["direction"] = directions
    simplified["modes"] = modes_arr
    simplified["link_type"] = link_types
    simplified["name"] = names
    simplified["speed_ab"] = speed_ab
    simplified["speed_ba"] = speed_ba
    simplified["lanes_ab"] = lanes_ab
    simplified["lanes_ba"] = lanes_ba
    simplified[_SOURCE_ID_COL] = primary_source_ids
    simplified[_PROVENANCE_OUT_COL] = provenance
    simplified[_SOURCE_REFS_TMP_COL] = source_refs

    if metrics is not None:
        metrics["transfer_time_s"] = perf_counter() - t0
        metrics["raw_hit_counts"] = raw_hit_counts
        metrics["reduced_candidate_counts"] = reduced_counts
        metrics["bearing_direct_counts"] = bearing_counts
        metrics["fallback_alignment_counts"] = fallback_counts


# Highway classes grouped by function. Modes are only inherited between
# originals that fall in the same functional family as the simplified link's
# nearest original, which stops e.g. footway/cycleway modes bleeding onto roads.
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


def _link_type_family(link_type: str):
    lt = (link_type or "").lower()
    for family in _LINK_TYPE_FAMILIES:
        if lt in family:
            return family
    return None


def _link_type_compatible(reference: str, candidate: str) -> bool:
    """Whether ``candidate`` may donate attributes to a link classed ``reference``.

    Same link type is always compatible. Otherwise both must belong to the same
    functional family. Unknown/unclassified types fall back to permissive so we
    never drop the only available candidate.
    """
    if reference == candidate:
        return True
    ref_family = _link_type_family(reference)
    cand_family = _link_type_family(candidate)
    if ref_family is None or cand_family is None:
        return True
    return ref_family is cand_family


def _geometry_summary(geom) -> dict:
    coords = geom.coords
    start = coords[0]
    end = coords[-1]
    return {
        "start": start,
        "end": end,
        "bearing": bearing_degrees(start, end),
        "straightness": line_straightness(geom),
        "length": float(geom.length),
    }


def _reduce_candidates_by_overlap(simplified_geom, orig_geoms, compatible: list[int]) -> list[tuple[int, float]]:
    if not compatible:
        return []
    # Real line/buffer intersections for every candidate were measurably expensive
    # in benchmarking. For ranking purposes we only need a cheap proxy that keeps
    # the plausible corridor candidates near the top.
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
    return [(oidx, dist) for oidx, _intersects, dist, _endpoint_cost in scored[: min(4, len(scored))]]


def _classify_candidates(
    *,
    reduced: list[tuple[int, float]],
    simp_geom,
    simp_summary: dict,
    orig_geoms,
    orig_straightness,
    orig_dir,
    orig_source_ids,
    metrics: dict | None,
):
    fwd_candidates = []
    bwd_candidates = []
    contributing_oidx: list[int] = []
    bearing_direct = 0
    fallback = 0

    for oidx, dist in reduced:
        orientation = _classify_orientation_fast(simp_summary, orig_geoms[oidx], orig_straightness[oidx])
        if orientation is None:
            fallback += 1
            orientation = _is_forward_aligned(simp_geom, orig_geoms[oidx], metrics=metrics)
        else:
            bearing_direct += 1

        d = int(orig_dir[oidx])
        base_id = orig_source_ids[oidx]
        contributing_oidx.append(int(oidx))

        if d == 0:
            if orientation:
                fwd_candidates.append((f"{base_id}::ab", dist))
                bwd_candidates.append((f"{base_id}::ba", dist))
            else:
                fwd_candidates.append((f"{base_id}::ba", dist))
                bwd_candidates.append((f"{base_id}::ab", dist))
        elif d == 1:
            if orientation:
                fwd_candidates.append((f"{base_id}::ab", dist))
            else:
                bwd_candidates.append((f"{base_id}::ab", dist))
        elif d == -1:
            if orientation:
                bwd_candidates.append((f"{base_id}::ba", dist))
            else:
                fwd_candidates.append((f"{base_id}::ba", dist))

    return fwd_candidates, bwd_candidates, contributing_oidx, bearing_direct, fallback


def _classify_orientation_fast(simp_summary: dict, orig_geom, orig_straightness: float) -> bool | None:
    orig_summary = _geometry_summary(orig_geom)
    if simp_summary["straightness"] < _STRAIGHTNESS_THRESHOLD or orig_straightness < _STRAIGHTNESS_THRESHOLD:
        return None

    same_cost = shapely.Point(simp_summary["start"]).distance(shapely.Point(orig_summary["start"])) + shapely.Point(
        simp_summary["end"]
    ).distance(shapely.Point(orig_summary["end"]))
    rev_cost = shapely.Point(simp_summary["start"]).distance(shapely.Point(orig_summary["end"])) + shapely.Point(
        simp_summary["end"]
    ).distance(shapely.Point(orig_summary["start"]))
    bearing_fwd = angular_difference_degrees(simp_summary["bearing"], orig_summary["bearing"])
    bearing_rev = angular_difference_degrees(simp_summary["bearing"], (orig_summary["bearing"] + 180.0) % 360.0)

    if same_cost < rev_cost and bearing_fwd <= _BEARING_MAX_DIFF_DEGREES:
        return True
    if rev_cost < same_cost and bearing_rev <= _BEARING_MAX_DIFF_DEGREES:
        return False
    return None


def _is_forward_aligned(geom_a, geom_b, metrics: dict | None = None) -> bool:
    """Return whether ``geom_b`` flows roughly in the same direction as ``geom_a``."""
    return aligned_along_geometry(geom_a, geom_b, metrics=metrics)


def _dekink_endpoints_local(geom):
    if geom is None or getattr(geom, "is_empty", False):
        return geom
    coords = list(geom.coords)
    if len(coords) < 4:
        return geom

    coords = _prune_endpoint_kinks(coords, reverse=False)
    coords = _prune_endpoint_kinks(coords, reverse=True)
    return shapely.LineString(coords)


def _prune_endpoint_kinks(coords: list, *, reverse: bool) -> list:
    work = list(reversed(coords)) if reverse else list(coords)
    max_steps = min(_DEKINK_MAX_POINTS, len(work) - 3)

    # First: greedily remove sharp immediate spikes.
    for _ in range(max_steps):
        if len(work) < 4:
            break
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

    # Second: collapse a short endpoint approach chain if it is noticeably more
    # sinuous than the direct chord to the first stable interior point. This
    # targets staircase-like last sections such as the one observed in manual experiments.
    for idx in range(2, min(len(work) - 1, _DEKINK_MAX_POINTS + 1)):
        chain = work[: idx + 1]
        chain_len = shapely.LineString(chain).length
        chord_len = shapely.LineString([chain[0], chain[-1]]).length
        if chord_len <= 0 or chain_len > _DEKINK_MAX_ENDPOINT_LENGTH:
            break
        if chain_len / chord_len < 1.02:
            continue
        if idx + 1 < len(work):
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


def _unique_source_refs(candidates: list[tuple[str, float]]) -> list[str]:
    seen = set()
    ordered = []
    for source_ref, _dist in sorted(candidates, key=lambda item: item[1]):
        if source_ref in seen:
            continue
        seen.add(source_ref)
        ordered.append(source_ref)
    return ordered


def _ordered_source_ids(source_refs: list[str]) -> list[str]:
    source_ids = []
    for source_ref in source_refs:
        source_id, _sep, _suffix = source_ref.partition("::")
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


