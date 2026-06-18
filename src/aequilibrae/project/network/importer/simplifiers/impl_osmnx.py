import geopandas as gpd
import json
import logging
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import is_missing, to_jsonable
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.importer.utils import NODE_ID_START
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)

_PROVENANCE_OUT_COL = "source_ids"
_SOURCE_ID_COL = "source_id"


def run_osmnx_simplify(
    net: StagedNetwork,
    *,
    consolidate_tolerance=10.0,
) -> StagedNetwork:
    ox = require("osmnx", feature="OSMnx simplification")

    graph = net.to_graph()
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ImporterError("OSMnx simplifier received an empty graph")

    projected = ox.projection.project_graph(graph)
    simplified = ox.simplification.simplify_graph(projected, edge_attrs_differ=("link_type", "name"))
    if consolidate_tolerance:
        simplified = ox.simplification.consolidate_intersections(
            simplified,
            tolerance=float(consolidate_tolerance),
            rebuild_graph=True,
            dead_ends=True,
        )
    return _graph_to_staged(net, ox.projection.project_graph(simplified, to_crs="EPSG:4326"))


def _graph_to_staged(net: StagedNetwork, graph) -> StagedNetwork:
    src_attrs = _build_source_attr_map(net.links)
    osm_to_new = {nid: NODE_ID_START + i for i, nid in enumerate(graph.nodes)}
    node_rows = []
    for nid, data in graph.nodes(data=True):
        geom = data.get("geometry") or Point(data["x"], data["y"])
        node_rows.append({"node_id": osm_to_new[nid], "geometry": geom, "modes": _coerce_modes(data.get("modes"))})

    link_rows = []
    for link_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
        geom = data.get("geometry") or LineString(
            [(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])]
        )
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda p: p.length)

        source_ids = _source_ids_for_edge(data)
        primary = source_ids[0] if source_ids else str(link_id)
        primary_attrs = src_attrs.get(primary, {})
        link_rows.append(
            {
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
                _SOURCE_ID_COL: primary,
                _PROVENANCE_OUT_COL: _build_provenance(source_ids, src_attrs),
            }
        )

    if not link_rows:
        raise ImporterError("OSMnx simplification produced zero links")

    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
    links_out = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")
    used = set(links_out["a_node"]) | set(links_out["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used)].reset_index(drop=True)

    out = StagedNetwork(nodes=nodes_out, links=links_out, source_meta=dict(net.source_meta))
    out.validate()
    return out


def _coerce_modes(value) -> str:
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
    if _SOURCE_ID_COL not in links_gdf.columns:
        return {}
    skip = {
        "a_node",
        "b_node",
        "link_id",
        "geometry",
        "direction",
        "distance",
        _PROVENANCE_OUT_COL,
    }
    out = {}
    for rec in links_gdf.to_dict(orient="records"):
        attrs = {}
        for col, val in rec.items():
            if is_missing(val) or col in skip or str(col).startswith("_"):
                continue
            attrs[str(col)] = to_jsonable(val)
        out[str(rec[_SOURCE_ID_COL])] = attrs
    return out


def _source_ids_for_edge(data: dict) -> list:
    raw = data.get(_SOURCE_ID_COL)
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
        modes = src_attrs.get(sid, {}).get("modes")
        if isinstance(modes, str):
            chars.update(modes)
    if not chars and isinstance(fallback, str):
        chars.update(fallback)
    return "".join(sorted(chars)) or "c"


def _aggregate_distance(geom, data: dict) -> float:
    length = data.get("length")
    if length is not None:
        return float(length)
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    return float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])
