"""OSMnx-based simplifier implementation.

Pipeline:
  1. project the staged network's MultiDiGraph to auto-UTM
  2. ``ox.simplification.simplify_graph`` (degree-2 interstitial collapse)
  3. ``ox.simplification.consolidate_intersections(tolerance=…)``
  4. project back to EPSG:4326
  5. rebuild the staged network, with the merged source ids forming the
     per-edge ``source_id_list`` dict-of-dicts in ``other_attributes``.
"""

import json
import logging

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.utils.optional_dependency import require

from ..schema.attributes import is_missing, to_jsonable
from ..staged_network import StagedNetwork

logger = logging.getLogger(__name__)


_PROVENANCE_OUT_COL = "source_id_list"
_PROVENANCE_PRIMARY = "source_id"
_NODE_START = 10000


def run_osmnx_simplify(
    net: StagedNetwork,
    *,
    consolidate_tolerance=10.0,
    edge_attr_aggs=None,
) -> StagedNetwork:
    """Apply OSMnx ``simplify_graph`` + ``consolidate_intersections``."""
    ox = require("osmnx", feature="OSMnx simplification")

    g = net.to_graph()
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        logger.warning("OSMnx simplifier received an empty graph; returning unchanged")
        return net

    g_proj = ox.projection.project_graph(g)
    g_simp = ox.simplification.simplify_graph(
        g_proj,
        edge_attrs_differ=("link_type", "name"),
    )

    if consolidate_tolerance is not None and consolidate_tolerance > 0:
        g_simp = ox.simplification.consolidate_intersections(
            g_simp,
            tolerance=float(consolidate_tolerance),
            rebuild_graph=True,
            dead_ends=True,
        )

    g_geo = ox.projection.project_graph(g_simp, to_crs="EPSG:4326")
    return _graph_to_staged(net, g_geo)


def _graph_to_staged(net: StagedNetwork, g) -> StagedNetwork:
    """Build a staged network from a simplified MultiDiGraph, reconstructing provenance."""
    source_attr_map = _build_source_attr_map(net.links)

    # ---- Build nodes
    node_rows = []
    new_id = _NODE_START
    osm_to_new: dict = {}
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
        node_rows.append({
            "node_id": new_node_id,
            "geometry": geom,
            "modes": data.get("modes", "c"),
        })
    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

    # ---- Build links with provenance
    link_rows = []
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

        source_ids = _source_ids_for_edge(data)
        provenance = _build_provenance(source_ids, source_attr_map)
        primary_id = source_ids[0] if source_ids else str(next_link_id)
        primary_attrs = source_attr_map.get(primary_id, {})

        link_rows.append({
            "link_id": next_link_id,
            "a_node": osm_to_new[u],
            "b_node": osm_to_new[v],
            "direction": int(data.get("direction", 0)),
            "modes": _aggregate_modes(source_ids, source_attr_map, fallback=data.get("modes", "c")),
            "link_type": primary_attrs.get("link_type") or data.get("link_type") or "unknown",
            "distance": _aggregate_distance(geom, data),
            "geometry": geom,
            "name": primary_attrs.get("name") or data.get("name"),
            "speed_ab": primary_attrs.get("speed_ab"),
            "speed_ba": primary_attrs.get("speed_ba"),
            "lanes_ab": primary_attrs.get("lanes_ab"),
            "lanes_ba": primary_attrs.get("lanes_ba"),
            _PROVENANCE_PRIMARY: primary_id,
            _PROVENANCE_OUT_COL: provenance,
        })
        next_link_id += 1

    if not link_rows:
        logger.warning("OSMnx simplification produced zero links; returning original staged network")
        return net

    links_out = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")

    used = set(links_out["a_node"]).union(links_out["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used)].reset_index(drop=True)

    out = StagedNetwork(
        nodes=nodes_out,
        links=links_out,
        source_meta=dict(net.source_meta),
    )
    out.validate()
    return out


def _build_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    """Build a lookup of pre-simplification ``_source_id`` → attribute dict."""
    if "_source_id" not in links_gdf.columns:
        return {}
    out: dict = {}
    skip = {"a_node", "b_node", "link_id", "geometry", "modes",
            "direction", "link_type", "distance",
            "speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "name",
            "_source_id", _PROVENANCE_OUT_COL, _PROVENANCE_PRIMARY}
    routing_fields = {"link_type", "name", "speed_ab", "speed_ba", "lanes_ab", "lanes_ba"}
    for _, row in links_gdf.iterrows():
        src = str(row["_source_id"])
        attrs = {}
        for col, val in row.items():
            if col in skip or str(col).startswith("_"):
                # routing-relevant typed fields are recorded into the provenance dict too
                if col in routing_fields and not is_missing(val):
                    attrs[col] = to_jsonable(val)
                continue
            if is_missing(val):
                continue
            attrs[str(col)] = to_jsonable(val)
        out[src] = attrs
    return out


def _source_ids_for_edge(data: dict) -> list:
    """Extract the ordered list of source ids attached to a simplified edge."""
    raw = data.get("_source_id")
    if raw is None:
        raw = data.get("merged_edges") or data.get("osmid")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(v) for v in raw]
    return [str(raw)]


def _build_provenance(source_ids: list, source_attr_map: dict):
    """Build the per-edge ``source_id_list`` JSON dict-of-dicts."""
    payload = {}
    for src in source_ids:
        payload[src] = source_attr_map.get(src, {})
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"), default=str)


def _aggregate_modes(source_ids: list, src_map: dict, fallback: str) -> str:
    out: set = set()
    for src in source_ids:
        m = src_map.get(src, {}).get("modes")
        if isinstance(m, str):
            out.update(m)
    if not out and isinstance(fallback, str):
        out.update(fallback)
    return "".join(sorted(out)) or "c"


def _aggregate_distance(geom, data: dict) -> float:
    """Compute the link distance in metres."""
    if data.get("length") is not None:
        return float(data["length"])
    if geom is None:
        return 0.0
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    return float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])
