import geopandas as gpd
import json
import numpy as np
import logging
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import is_missing, to_jsonable
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.importer.utils import NODE_ID_START
from aequilibrae.utils.optional_dependency import require
import pandas as pd

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

    # ---- Vectorised link construction ----
    node_xy = {nid: (d["x"], d["y"]) for nid, d in graph.nodes(data=True)}
    df = pd.DataFrame(list(graph.edges(data=True)), columns=["_u", "_v", "_data"])
    if len(df) == 0:
        raise ImporterError("OSMnx simplification produced zero links")
    df = df.join(pd.json_normalize(df["_data"]))

    # Core IDs
    df["link_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["a_node"] = df["_u"].map(osm_to_new).astype(np.int64)
    df["b_node"] = df["_v"].map(osm_to_new).astype(np.int64)

    # Direction: resolve lists produced by OSMnx edge merging
    if "direction" in df.columns:
        df["direction"] = df["direction"].apply(
            lambda v: v[0] if isinstance(v, list) and len(set(v)) == 1 else 0 if isinstance(v, list) else v
        ).fillna(0).astype(int)
    else:
        df["direction"] = 0

    # Geometry: fill missing with straight line, resolve MultiLineString
    def _resolve_geom(row):
        g = row.get("geometry")
        if g is None or (hasattr(g, "is_empty") and g.is_empty):
            g = LineString([node_xy[row["_u"]], node_xy[row["_v"]]])
        if isinstance(g, MultiLineString):
            g = max(g.geoms, key=lambda p: p.length)
        return g

    df["geometry"] = df.apply(_resolve_geom, axis=1)

    # Source IDs and primary source
    _sid = _SOURCE_ID_COL
    if _sid in df.columns:
        df["_source_ids"] = df[_sid].apply(
            lambda v: [str(x) for x in v] if isinstance(v, (list, tuple, set))
            else [str(v)] if v is not None and not (isinstance(v, float) and np.isnan(v)) else []
        )
        df[_sid] = [ids[0] if ids else str(lid) for ids, lid in zip(df["_source_ids"], df["link_id"])]
    else:
        df["_source_ids"] = [[] for _ in range(len(df))]
        df[_sid] = df["link_id"].astype(str)

    # Look up primary attrs from the pre-simplification network
    pa = df[_sid].map(lambda s: src_attrs.get(s, {}))

    # Modes: aggregate from source attrs, fall back to edge value (resolve lists first)
    edge_modes = df["modes"].apply(_coerce_modes) if "modes" in df.columns else pd.Series("c", index=df.index)
    df["modes"] = [_aggregate_modes(sids, src_attrs, m) for sids, m in zip(df["_source_ids"], edge_modes)]

    # Link type: primary_attrs > edge data > "unknown"
    pa_lt = pa.apply(lambda a: a.get("link_type"))
    edge_lt = df["link_type"] if "link_type" in df.columns else pd.Series(dtype=object, index=df.index)
    df["link_type"] = pa_lt.combine_first(edge_lt).fillna("unknown")

    # Distance: prefer pre-computed length, else measure from geometry
    if "length" in df.columns:
        df["distance"] = df["length"].apply(
            lambda v: float(sum(v)) if isinstance(v, list) else float(v) if pd.notna(v) else np.nan
        )
    else:
        df["distance"] = np.nan
    need_dist = df["distance"].isna()
    if need_dist.any():
        gs = gpd.GeoSeries(df.loc[need_dist, "geometry"].values, crs="EPSG:4326")
        df.loc[need_dist, "distance"] = gs.to_crs(gs.estimate_utm_crs()).length.values

    # Name: primary_attrs > edge data
    pa_name = pa.apply(lambda a: a.get("name"))
    edge_name = df["name"] if "name" in df.columns else pd.Series(dtype=object, index=df.index)
    df["name"] = pa_name.combine_first(edge_name)

    # Speed and lane columns from primary attrs
    for col in ("speed_ab", "speed_ba", "lanes_ab", "lanes_ba"):
        df[col] = pa.apply(lambda a, c=col: a.get(c))

    # Provenance
    df[_PROVENANCE_OUT_COL] = df["_source_ids"].apply(lambda ids: _build_provenance(ids, src_attrs))

    # Select output columns
    out_cols = ["link_id", "a_node", "b_node", "direction", "modes", "link_type",
                "distance", "geometry", "name", "speed_ab", "speed_ba", "lanes_ab", "lanes_ba",
                _sid, _PROVENANCE_OUT_COL]
    links_out = gpd.GeoDataFrame(df[[c for c in out_cols if c in df.columns]], geometry="geometry", crs="EPSG:4326")

    # ---- Old scalar loop (kept for verification) ----
    # link_rows = []
    # for link_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
    #     geom = data.get("geometry") or LineString(
    #         [(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])]
    #     )
    #     if isinstance(geom, MultiLineString):
    #         geom = max(geom.geoms, key=lambda p: p.length)
    #
    #     source_ids = _source_ids_for_edge(data)
    #     primary = source_ids[0] if source_ids else str(link_id)
    #     primary_attrs = src_attrs.get(primary, {})
    #     link_rows.append(
    #         {
    #             "link_id": link_id,
    #             "a_node": osm_to_new[u],
    #             "b_node": osm_to_new[v],
    #             "direction": int(data.get("direction", 0)),
    #             "modes": _aggregate_modes(source_ids, src_attrs, data.get("modes", "c")),
    #             "link_type": primary_attrs.get("link_type") or data.get("link_type") or "unknown",
    #             "distance": _aggregate_distance(geom, data),
    #             "geometry": geom,
    #             "name": primary_attrs.get("name") or data.get("name"),
    #             "speed_ab": primary_attrs.get("speed_ab"),
    #             "speed_ba": primary_attrs.get("speed_ba"),
    #             "lanes_ab": primary_attrs.get("lanes_ab"),
    #             "lanes_ba": primary_attrs.get("lanes_ba"),
    #             _SOURCE_ID_COL: primary,
    #             _PROVENANCE_OUT_COL: _build_provenance(source_ids, src_attrs),
    #         }
    #     )
    # if not link_rows:
    #     raise ImporterError("OSMnx simplification produced zero links")
    # links_out = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")

    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
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
