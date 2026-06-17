"""OSMnx-based simplifier implementation.

Pipeline (plan §7):
  1. project the IR's MultiDiGraph to auto-UTM (no user override per §1.3 rule 4)
  2. ``ox.simplification.simplify_graph`` (degree-2 interstitial collapse)
  3. ``ox.simplification.consolidate_intersections(tolerance=…)``
     (handles dual carriageways, roundabouts, complex junctions)
  4. project back to EPSG:4326
  5. rebuild the IR, with the merged source ids forming the per-edge
     ``source_id_list`` dict-of-dicts in ``other_attributes`` (plan §4.4.1).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.utils.optional_dependency import require

from ..ir import RoutableNetwork

logger = logging.getLogger(__name__)


_PROVENANCE_OUT_COL = "source_id_list"
_PROVENANCE_PRIMARY = "source_id"
_NODE_START = 10000


def run_osmnx_simplify(
    net: RoutableNetwork,
    *,
    consolidate_tolerance: float | None = 10.0,
    edge_attr_aggs: dict[str, Any] | None = None,
) -> RoutableNetwork:
    """Apply OSMnx ``simplify_graph`` + ``consolidate_intersections``."""
    ox = require("osmnx", feature="OSMnx simplification")

    g = net.to_multidigraph()
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        logger.warning("OSMnx simplifier received an empty graph; returning unchanged")
        return net

    # ---- Project to auto-UTM (no user override per plan §1.3 rule 4)
    g_proj = ox.projection.project_graph(g)

    # ---- Simplify: collapse degree-2 interstitial chains
    try:
        g_simp = ox.simplification.simplify_graph(
            g_proj,
            edge_attrs_differ=("link_type", "name"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"OSMnx simplify_graph failed ({exc}); skipping simplification")
        return net

    # ---- Optional consolidation of nearby intersections
    if consolidate_tolerance is not None and consolidate_tolerance > 0:
        try:
            g_simp = ox.simplification.consolidate_intersections(
                g_simp,
                tolerance=float(consolidate_tolerance),
                rebuild_graph=True,
                dead_ends=True,
            )
        except Exception as exc:
            logger.warning(
                f"OSMnx consolidate_intersections failed ({exc}); "
                "continuing with simplify-only result"
            )

    # ---- Project back to EPSG:4326
    g_geo = ox.projection.project_graph(g_simp, to_crs="EPSG:4326")

    # ---- Convert back to IR. We do this by hand (not via
    # RoutableNetwork.from_multidigraph) so we can build the per-merged-edge
    # dict-of-dicts ``source_id_list`` from the original IR.
    return _multidigraph_to_ir(net, g_geo)


def _multidigraph_to_ir(net: RoutableNetwork, g) -> RoutableNetwork:
    """Build an IR from a simplified MultiDiGraph, reconstructing provenance."""
    # Cache the source-edge attribute set for fast lookup by _source_id
    source_attr_map = _build_source_attr_map(net.links)

    # ---- Build nodes
    node_rows = []
    new_id = _NODE_START
    osm_to_new: dict[int, int] = {}
    for nid, data in g.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                continue
            geom = Point(x, y)
        new_node_id = new_id
        new_id += 1
        osm_to_new[nid] = new_node_id
        row = {
            "node_id": new_node_id,
            "geometry": geom,
            "modes": data.get("modes", "c"),
        }
        node_rows.append(row)
    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

    # ---- Build links with provenance
    link_rows: list[dict] = []
    next_link_id = 1
    for u, v, _key, data in g.edges(keys=True, data=True):
        if u not in osm_to_new or v not in osm_to_new:
            continue
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([
                (g.nodes[u].get("x"), g.nodes[u].get("y")),
                (g.nodes[v].get("x"), g.nodes[v].get("y")),
            ])
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda p: p.length)

        # Pull source-id(s) from the simplified edge: osmnx puts them in
        # 'osmid' (single int) or 'osmid' (list). We mapped the IR's
        # _source_id into the edge attributes when calling to_multidigraph.
        source_ids = _source_ids_for_edge(data)
        provenance = _build_provenance(source_ids, source_attr_map)
        primary_id = source_ids[0] if source_ids else str(next_link_id)
        primary_attrs = source_attr_map.get(primary_id, {})

        row = {
            "link_id": next_link_id,
            "a_node": osm_to_new[u],
            "b_node": osm_to_new[v],
            "direction": int(data.get("direction", 0)),
            "modes": _aggregate_modes(source_ids, source_attr_map, fallback=data.get("modes", "c")),
            "link_type": primary_attrs.get("link_type") or data.get("link_type") or "unknown",
            "distance": _aggregate_distance(geom, data),
            "geometry": geom,
            # Carry the lifted "primary" attributes as free-form IR columns so
            # they end up in real DB columns where they match the schema,
            # otherwise they land in other_attributes alongside the dict-of-dicts.
            "name": primary_attrs.get("name") or data.get("name"),
            "speed_ab": primary_attrs.get("speed_ab"),
            "speed_ba": primary_attrs.get("speed_ba"),
            "lanes_ab": primary_attrs.get("lanes_ab"),
            "lanes_ba": primary_attrs.get("lanes_ba"),
            _PROVENANCE_PRIMARY: primary_id,
            _PROVENANCE_OUT_COL: provenance,
        }
        link_rows.append(row)
        next_link_id += 1

    if not link_rows:
        logger.warning("OSMnx simplification produced zero links; returning original IR")
        return net

    links_out = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")

    # Drop orphan nodes
    used = set(links_out["a_node"]).union(links_out["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used)].reset_index(drop=True)

    ir = RoutableNetwork(
        nodes=nodes_out,
        links=links_out,
        source_meta=dict(net.source_meta),
    )
    ir.validate()
    return ir


def _build_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict[str, dict]:
    """Build a lookup of pre-simplification ``_source_id`` → attribute dict.

    The dict-of-dicts provenance written to ``other_attributes`` uses this
    mapping so every merged edge's tag set is preserved verbatim.
    """
    if "_source_id" not in links_gdf.columns:
        return {}
    out: dict[str, dict] = {}
    skip = {"a_node", "b_node", "link_id", "geometry", "modes",
            "direction", "link_type", "distance",
            "speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "name",
            "_source_id", _PROVENANCE_OUT_COL, _PROVENANCE_PRIMARY}
    for _, row in links_gdf.iterrows():
        src = str(row["_source_id"])
        attrs = {}
        for col, val in row.items():
            if col in skip or str(col).startswith("_"):
                # but we DO want to record the routing-relevant typed fields so
                # the merged-edge dict knows what each source carried
                if col in ("link_type", "name", "speed_ab", "speed_ba",
                           "lanes_ab", "lanes_ba"):
                    if val is not None and not _is_nan(val):
                        attrs[col] = _json_safe(val)
                continue
            if val is None or _is_nan(val):
                continue
            attrs[str(col)] = _json_safe(val)
        out[src] = attrs
    return out


def _source_ids_for_edge(data: dict) -> list[str]:
    """Extract the ordered list of source ids attached to a simplified edge."""
    raw = data.get("_source_id")
    if raw is None:
        raw = data.get("merged_edges") or data.get("osmid")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(v) for v in raw]
    return [str(raw)]


def _build_provenance(source_ids: list[str], source_attr_map: dict[str, dict]) -> str:
    """Build the per-edge ``source_id_list`` JSON dict-of-dicts."""
    payload = {}
    for src in source_ids:
        payload[src] = source_attr_map.get(src, {})
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"), default=str)


def _aggregate_modes(source_ids: list[str], src_map: dict[str, dict], fallback: str) -> str:
    """Union the per-source-edge mode strings into the merged-edge modes."""
    out: set[str] = set()
    for src in source_ids:
        m = src_map.get(src, {}).get("modes")
        if isinstance(m, str):
            out.update(m)
    if not out and isinstance(fallback, str):
        out.update(fallback)
    return "".join(sorted(out)) or "c"


def _aggregate_distance(geom, data: dict) -> float:
    """Compute the link distance in metres.

    osmnx stores a ``length`` attribute in the projected CRS units after
    simplify_graph; we recompute from the EPSG:4326 geometry's auto-UTM
    projection to be safe.
    """
    if data.get("length") is not None:
        try:
            return float(data["length"])
        except (TypeError, ValueError):
            pass
    if geom is None:
        return 0.0
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    return float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])


# ---------- small helpers ----------

def _is_nan(value) -> bool:
    if isinstance(value, float):
        return value != value  # noqa: PLR0124
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value):
    if value is None or _is_nan(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # pragma: no cover
            pass
    return str(value)
