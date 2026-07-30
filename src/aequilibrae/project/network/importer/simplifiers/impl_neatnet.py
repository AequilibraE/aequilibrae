import geopandas as gpd
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


def run_neatnet_simplify(
    net: StagedNetwork,
    *,
    exclusion_mask=None,
    consolidate_tolerance: float | None = _DEFAULT_CONSOLIDATE_TOLERANCE,
    simplification_factor: float = 2.0,
    min_dangle_length: float = 20.0,
) -> StagedNetwork:
    require("neatnet", feature="neatnet simplification")

    import neatnet

    warnings.warn(_DUAL_CARRIAGEWAY_WARNING, UserWarning, stacklevel=2)

    if len(net.links) == 0:
        return net

    # ``None`` disables the optional consolidation pass in the OSMnx simplifier,
    # but node consolidation is integral to neatify, so fall back to its default.
    if consolidate_tolerance is None:
        consolidate_tolerance = _DEFAULT_CONSOLIDATE_TOLERANCE

    edges = net.links.copy()
    utm = edges.geometry.estimate_utm_crs()
    geom_only = gpd.GeoDataFrame(geometry=edges.geometry, crs=edges.crs).to_crs(utm)

    neatify_kwargs = {
        "consolidation_tolerance": float(consolidate_tolerance),
        "simplification_factor": float(simplification_factor),
        "min_dangle_length": float(min_dangle_length),
    }
    if exclusion_mask is not None:
        neatify_kwargs["exclusion_mask"] = exclusion_mask.to_crs(utm).geometry

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neatnet")
        simplified = neatnet.neatify(geom_only, **neatify_kwargs).to_crs("EPSG:4326")

    return _gdf_to_staged(simplified, original_links=edges, source_meta=net.source_meta)


def _gdf_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    original_links: gpd.GeoDataFrame,
    source_meta: dict,
) -> StagedNetwork:
    edges = edges_gdf.copy().reset_index(drop=True)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)

    _transfer_attributes(edges, original_links)
    edges["geometry"] = [_dekink_endpoints_local(geom) for geom in edges.geometry]

    endpoints, a_nodes, b_nodes, _next_id = _build_endpoint_index(edges.geometry)
    edges["a_node"] = a_nodes
    edges["b_node"] = b_nodes
    edges["distance"] = compute_lengths(edges.geometry).to_numpy()

    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(endpoints.keys()),
            "geometry": [Point(x, y) for x, y in endpoints.values()],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes["modes"] = compute_node_modes(nodes["node_id"].to_numpy(), edges, fallback="c")

    return StagedNetwork(nodes=nodes, links=edges, source_meta=source_meta)


_BUFFER_DIST = 25.0  # metres – search radius for matching original edges


def _build_endpoint_index(geoms):
    coords, indices = shapely.get_coordinates(geoms, return_index=True)
    last_pos = np.searchsorted(indices, np.arange(len(geoms)), side="right") - 1
    first_pos = np.searchsorted(indices, np.arange(len(geoms)), side="left")
    starts = coords[first_pos]
    ends = coords[last_pos]

    node_lookup = {}
    a_nodes = np.empty(len(geoms), dtype=np.int64)
    b_nodes = np.empty(len(geoms), dtype=np.int64)
    next_id = NODE_ID_START
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for arr, target in ((start, a_nodes), (end, b_nodes)):
            key = (round(float(arr[0]), 7), round(float(arr[1]), 7))
            nid = node_lookup.get(key)
            if nid is None:
                nid = next_id
                node_lookup[key] = nid
                next_id += 1
            target[i] = nid
    endpoints = {nid: key for key, nid in node_lookup.items()}
    return endpoints, a_nodes, b_nodes, next_id


def _transfer_attributes(simplified: gpd.GeoDataFrame, original: gpd.GeoDataFrame) -> None:
    """Match each simplified edge to nearby originals and aggregate attributes."""
    utm = simplified.geometry.estimate_utm_crs()
    simp_geoms = simplified.geometry.to_crs(utm).values
    orig_geoms = original.geometry.to_crs(utm).values
    src_attrs = build_source_attr_map(original)
    oriented_src_attrs = build_oriented_source_attr_map(original)

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

    orig_dir = original["direction"].to_numpy()
    orig_modes = original["modes"].to_numpy()
    orig_lt = original["link_type"].to_numpy()
    orig_name = original["name"].to_numpy()
    orig_source_ids = original[SOURCE_ID_COL].astype(str).to_numpy()
    orig_straightness = np.array([line_straightness(g) for g in orig_geoms], dtype=float)

    for i in range(n):
        sg = simp_geoms[i]
        hits = tree.query(sg.buffer(_BUFFER_DIST))
        if len(hits) == 0:
            hits = np.array([tree.nearest(sg)])

        nearest_oidx = int(tree.nearest(sg))
        nearest_lt = str(orig_lt[nearest_oidx])
        simp_summary = _geometry_summary(sg)

        compatible = [int(oidx) for oidx in hits if _link_type_compatible(nearest_lt, str(orig_lt[oidx]))]
        reduced = _reduce_candidates_by_overlap(sg, orig_geoms, compatible)

        fwd_candidates, bwd_candidates, contributing_oidx = _classify_candidates(
            reduced=reduced,
            simp_geom=sg,
            simp_summary=simp_summary,
            orig_geoms=orig_geoms,
            orig_straightness=orig_straightness,
            orig_dir=orig_dir,
            orig_source_ids=orig_source_ids,
        )

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

        ordered_source_ids = _ordered_source_ids(fwd_candidates + bwd_candidates)
        primary_source_ids[i] = ordered_source_ids[0] if ordered_source_ids else orig_source_ids[nearest_oidx]
        provenance[i] = build_provenance(ordered_source_ids, src_attrs)

        sources = contributing_oidx or [nearest_oidx]
        all_modes = set().union(*(orig_modes[o] for o in sources if isinstance(orig_modes[o], str)), set())
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
    simplified[SOURCE_ID_COL] = primary_source_ids
    simplified[PROVENANCE_OUT_COL] = provenance


# Highway classes grouped by function. Modes are only inherited between
# originals that fall in the same functional family as the simplified link's
# nearest original, which stops e.g. footway/cycleway modes bleeding onto roads.
_LINK_TYPE_FAMILIES = (
    {"motorway", "motorway_link", "trunk", "trunk_link"},
    {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
     "unclassified", "residential", "living_street", "service", "road", "busway", "bus_guideway"},
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
    }


def _reduce_candidates_by_overlap(simplified_geom, orig_geoms, compatible: list[int]) -> list[tuple[int, float]]:
    if not compatible:
        return []
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
):
    fwd_candidates = []
    bwd_candidates = []
    contributing_oidx: list[int] = []

    for oidx, dist in reduced:
        orientation = _classify_orientation_fast(simp_summary, orig_geoms[oidx], orig_straightness[oidx])
        if orientation is None:
            orientation = aligned_along_geometry(simp_geom, orig_geoms[oidx])

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

    return fwd_candidates, bwd_candidates, contributing_oidx


def _classify_orientation_fast(simp_summary: dict, orig_geom, orig_straightness: float) -> bool | None:
    if simp_summary["straightness"] < _STRAIGHTNESS_THRESHOLD or orig_straightness < _STRAIGHTNESS_THRESHOLD:
        return None
    orig_summary = _geometry_summary(orig_geom)
    (sx0, sy0), (sx1, sy1) = simp_summary["start"], simp_summary["end"]
    (ox0, oy0), (ox1, oy1) = orig_summary["start"], orig_summary["end"]

    same_cost = math.hypot(sx0 - ox0, sy0 - oy0) + math.hypot(sx1 - ox1, sy1 - oy1)
    rev_cost = math.hypot(sx0 - ox1, sy0 - oy1) + math.hypot(sx1 - ox0, sy1 - oy0)
    bearing_fwd = angular_difference_degrees(simp_summary["bearing"], orig_summary["bearing"])
    bearing_rev = angular_difference_degrees(simp_summary["bearing"], (orig_summary["bearing"] + 180.0) % 360.0)

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
    # targets staircase-like last sections near intersections.
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


def _ordered_source_ids(candidates: list[tuple[str, float]]) -> list[str]:
    """Base source ids from ``(source_ref, distance)`` candidates, nearest first, de-duplicated."""
    source_ids = []
    for source_ref, _dist in sorted(candidates, key=lambda item: item[1]):
        source_id = source_ref.partition("::")[0]
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


