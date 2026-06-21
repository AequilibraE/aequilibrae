import geopandas as gpd
import logging
import numpy as np
import shapely
from shapely.geometry import Point

from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.importer.utils import NODE_ID_START, compute_node_modes
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)


def run_neatnet_simplify(net: StagedNetwork, **_) -> StagedNetwork:
    require("neatnet", feature="neatnet simplification")
    import warnings

    import neatnet

    if len(net.links) == 0:
        return net

    edges = net.links.copy()
    utm = edges.geometry.estimate_utm_crs()

    # Strip all non-geometry columns before passing to neatnet so it cannot
    # mangle AequilibraE-specific attributes (direction, modes, …) during
    # edge merging.  Keep only the geometry for the simplifier.
    geom_only = gpd.GeoDataFrame(geometry=edges.geometry, crs=edges.crs).to_crs(utm)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="neatnet")
        simplified = neatnet.neatify(geom_only).to_crs("EPSG:4326")

    return _gdf_to_staged(simplified, original_links=edges, source_meta=net.source_meta)


def _gdf_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    original_links: gpd.GeoDataFrame,
    source_meta: dict,
) -> StagedNetwork:
    edges = edges_gdf.copy().reset_index(drop=True)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    # ---- Build topology (nodes from endpoints) ----
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

    # ---- Spatial matching: recover attributes from original network ----
    _transfer_attributes(edges, original_links)

    # ---- Distance (always recompute from geometry) ----
    utm = edges.geometry.estimate_utm_crs()
    edges["distance"] = edges.geometry.to_crs(utm).length.astype(float)

    # ---- Nodes ----
    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(endpoints.values()),
            "geometry": [Point(x, y) for x, y in endpoints],
            "modes": "c",  # placeholder, overwritten below
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes["modes"] = compute_node_modes(nodes["node_id"].to_numpy(), edges, fallback="c")

    return StagedNetwork(nodes=nodes, links=edges, source_meta=source_meta)


_BUFFER_DIST = 25.0  # metres – search radius for matching original edges


def _transfer_attributes(simplified: gpd.GeoDataFrame, original: gpd.GeoDataFrame) -> None:
    """Match each simplified edge to nearby originals and aggregate attributes.

    Dual carriageways (two opposing one-way links) are detected and merged
    into a single bidirectional link with correct per-direction speed and
    lane values.
    """
    utm = simplified.geometry.estimate_utm_crs()
    simp_geoms = simplified.geometry.to_crs(utm).values
    orig_geoms = original.geometry.to_crs(utm).values

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

    # Pre-extract original columns as arrays for fast indexing
    orig_dir = original["direction"].values if "direction" in original.columns else np.zeros(len(original), dtype=int)
    orig_modes = original["modes"].values if "modes" in original.columns else np.full(len(original), "c")
    orig_lt = original["link_type"].values if "link_type" in original.columns else np.full(len(original), "unknown")
    orig_name = original["name"].values if "name" in original.columns else np.full(len(original), None, dtype=object)
    orig_spd_ab = original["speed_ab"].values if "speed_ab" in original.columns else np.full(len(original), None, dtype=object)
    orig_spd_ba = original["speed_ba"].values if "speed_ba" in original.columns else np.full(len(original), None, dtype=object)
    orig_ln_ab = original["lanes_ab"].values if "lanes_ab" in original.columns else np.full(len(original), None, dtype=object)
    orig_ln_ba = original["lanes_ba"].values if "lanes_ba" in original.columns else np.full(len(original), None, dtype=object)

    for i in range(n):
        sg = simp_geoms[i]
        hits = tree.query(sg.buffer(_BUFFER_DIST))
        if len(hits) == 0:
            hits = np.array([tree.nearest(sg)])

        # Classify each hit as contributing forward (a→b) or backward (b→a)
        # flow relative to the *simplified* edge, based on geometric alignment
        # and the original link's direction.
        #   aligned + dir=1  → forward flow    aligned + dir=-1 → backward flow
        #   anti    + dir=1  → backward flow   anti    + dir=-1 → forward flow
        #   dir=0            → both flows (regardless of alignment)
        fwd_candidates = []  # (orig_idx, distance, aligned)
        bwd_candidates = []  # (orig_idx, distance, aligned)

        for oidx in hits:
            aligned = _is_forward_aligned(sg, orig_geoms[oidx])
            d = int(orig_dir[oidx])
            dist = float(sg.distance(orig_geoms[oidx]))

            if d == 0:
                fwd_candidates.append((oidx, dist, aligned))
                bwd_candidates.append((oidx, dist, aligned))
            elif d == 1:
                if aligned:
                    fwd_candidates.append((oidx, dist, True))
                else:
                    bwd_candidates.append((oidx, dist, False))
            elif d == -1:
                if aligned:
                    bwd_candidates.append((oidx, dist, True))
                else:
                    fwd_candidates.append((oidx, dist, False))

        has_fwd = len(fwd_candidates) > 0
        has_bwd = len(bwd_candidates) > 0

        if has_fwd and has_bwd:
            directions[i] = 0
        elif has_fwd:
            directions[i] = 1
        elif has_bwd:
            directions[i] = -1
        # else stays 0 (default)

        # Scalar attributes from the single nearest original
        nearest_oidx = int(tree.nearest(sg))
        link_types[i] = str(orig_lt[nearest_oidx])
        names[i] = orig_name[nearest_oidx]

        # Modes: union across all matched originals
        all_modes: set = set()
        for oidx in hits:
            m = orig_modes[oidx]
            if isinstance(m, str):
                all_modes.update(m)
        modes_arr[i] = "".join(sorted(all_modes)) or "c"

        # Speed / lanes from closest contributor in each direction.
        # When aligned:      ab(orig) → ab(simp),  ba(orig) → ba(simp)
        # When anti-aligned:  ba(orig) → ab(simp),  ab(orig) → ba(simp)
        if fwd_candidates:
            oidx, _, aligned = min(fwd_candidates, key=lambda t: t[1])
            if aligned:
                speed_ab[i] = orig_spd_ab[oidx]
                lanes_ab[i] = orig_ln_ab[oidx]
            else:
                speed_ab[i] = orig_spd_ba[oidx]
                lanes_ab[i] = orig_ln_ba[oidx]

        if bwd_candidates:
            oidx, _, aligned = min(bwd_candidates, key=lambda t: t[1])
            if aligned:
                speed_ba[i] = orig_spd_ba[oidx]
                lanes_ba[i] = orig_ln_ba[oidx]
            else:
                speed_ba[i] = orig_spd_ab[oidx]
                lanes_ba[i] = orig_ln_ab[oidx]

    simplified["direction"] = directions
    simplified["modes"] = modes_arr
    simplified["link_type"] = link_types
    simplified["name"] = names
    simplified["speed_ab"] = speed_ab
    simplified["speed_ba"] = speed_ba
    simplified["lanes_ab"] = lanes_ab
    simplified["lanes_ba"] = lanes_ba


def _is_forward_aligned(geom_a, geom_b) -> bool:
    """True if *geom_b* flows roughly in the same direction as *geom_a*.

    Compares the start→end direction vectors using a dot-product sign test.
    """
    ca = geom_a.coords
    cb = geom_b.coords
    va = (ca[-1][0] - ca[0][0], ca[-1][1] - ca[0][1])
    vb = (cb[-1][0] - cb[0][0], cb[-1][1] - cb[0][1])
    return (va[0] * vb[0] + va[1] * vb[1]) >= 0

