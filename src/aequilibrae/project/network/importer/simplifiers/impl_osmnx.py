"""OSMnx-based simplifier implementation.

Project to auto-UTM → ``simplify_graph`` → ``consolidate_intersections`` →
project back to EPSG:4326 → rebuild the staged network with per-merged-edge
``source_id_list`` in ``other_attributes``.
"""

import json
import logging

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.project.network.importer.schema.attributes import is_missing, to_jsonable
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

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
        g_proj, edge_attrs_differ=("link_type", "name")
    )
    if consolidate_tolerance:
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
    src_attrs = _build_source_attr_map(net.links)

    # ---- Nodes
    osm_to_new = {nid: _NODE_START + i for i, nid in enumerate(g.nodes)}
    node_rows = []
    for nid, data in g.nodes(data=True):
        geom = data.get("geometry") or Point(data["x"], data["y"])
        node_rows.append({
            "node_id": osm_to_new[nid],
            "geometry": geom,
            "modes": _coerce_modes(data.get("modes")),
        })
    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

    # ---- Links
    link_rows = []
    for link_id, (u, v, data) in enumerate(g.edges(data=True), start=1):
        geom = data.get("geometry") or LineString([
            (g.nodes[u]["x"], g.nodes[u]["y"]),
            (g.nodes[v]["x"], g.nodes[v]["y"]),
        ])
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda p: p.length)

        source_ids = _source_ids_for_edge(data)
        primary = source_ids[0] if source_ids else str(link_id)
        primary_attrs = src_attrs.get(primary, {})

        link_rows.append({
            "link_id": link_id,
            "a_node": osm_to_new[u],
            "b_node": osm_to_new[v],
            "direction": int(data.get("direction", 0)),
            "modes": _aggregate_modes(source_ids, src_attrs, data.get("modes", "c")),
            "link_type": primary_attrs.get("link_type") or data.get("link_type") or "unknown",
            "distance": _aggregate_distance(geom, data),
            "geometry": geom,
            "name": primary_attrs.get("name") or data.get("name"),
            "speed_ab": primary_attrs.get("speed_ab"),
            "speed_ba": primary_attrs.get("speed_ba"),
            "lanes_ab": primary_attrs.get("lanes_ab"),
            "lanes_ba": primary_attrs.get("lanes_ba"),
            _PROVENANCE_PRIMARY: primary,
            _PROVENANCE_OUT_COL: _build_provenance(source_ids, src_attrs),
        })

    if not link_rows:
        logger.warning("OSMnx simplification produced zero links; returning original staged network")
        return net

    links_out = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")
    used = set(links_out["a_node"]) | set(links_out["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used)].reset_index(drop=True)

    out = StagedNetwork(nodes=nodes_out, links=links_out, source_meta=dict(net.source_meta))
    out.validate()
    return out


def _coerce_modes(value) -> str:
    """OSMnx returns merged-node modes as a list; coerce to AequilibraE's char-string form."""
    if value is None:
        return "c"
    if isinstance(value, str):
        return value
    chars = set()
    for item in value:
        if isinstance(item, str):
            chars.update(item)
    return "".join(sorted(chars)) or "c"


def _build_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    """Lookup of pre-simplification ``_source_id`` → attribute dict (last write wins)."""
    if "_source_id" not in links_gdf.columns:
        return {}
    skip = {
        "a_node", "b_node", "link_id", "geometry", "modes",
        "direction", "link_type", "distance",
        "speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "name",
        "_source_id", _PROVENANCE_OUT_COL, _PROVENANCE_PRIMARY,
    }
    routing = {"link_type", "name", "speed_ab", "speed_ba", "lanes_ab", "lanes_ba"}
    out = {}
    for rec in links_gdf.to_dict(orient="records"):
        attrs = {}
        for col, val in rec.items():
            if is_missing(val):
                continue
            if col in skip or str(col).startswith("_"):
                if col in routing:
                    attrs[col] = to_jsonable(val)
                continue
            attrs[str(col)] = to_jsonable(val)
        out[str(rec["_source_id"])] = attrs
    return out


def _source_ids_for_edge(data: dict) -> list:
    raw = data.get("_source_id") or data.get("merged_edges") or data.get("osmid")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(v) for v in raw]
    return [str(raw)]


def _build_provenance(source_ids: list, src_attrs: dict):
    if not source_ids:
        return None
    payload = {sid: src_attrs.get(sid, {}) for sid in source_ids}
    return json.dumps(payload, separators=(",", ":"), default=str)


def _aggregate_modes(source_ids: list, src_attrs: dict, fallback) -> str:
    chars = set()
    for sid in source_ids:
        m = src_attrs.get(sid, {}).get("modes")
        if isinstance(m, str):
            chars.update(m)
    if not chars and isinstance(fallback, str):
        chars.update(fallback)
    return "".join(sorted(chars)) or "c"


def _aggregate_distance(geom, data: dict) -> float:
    """Distance in metres — prefer osmnx's pre-computed length, else project geometry."""
    length = data.get("length")
    if length is not None:
        return float(length)
    if geom is None:
        return 0.0
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    return float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])
