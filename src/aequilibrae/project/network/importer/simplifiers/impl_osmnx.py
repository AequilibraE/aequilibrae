import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import is_missing
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
    compute_lengths,
    compute_node_modes,
)
from aequilibrae.utils.optional_dependency import require

_SOURCE_REF_COL = "_source_ref"


def run_osmnx_simplify(
    net: StagedNetwork,
    *,
    consolidate_tolerance=10.0,
) -> StagedNetwork:
    ox = require("osmnx", feature="OSMnx simplification")

    graph = net.to_graph()
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ImporterError("OSMnx simplifier received an empty graph")

    graph.graph["simplified"] = True
    projected = ox.projection.project_graph(graph)
    projected.graph["simplified"] = False

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
    src_attrs = build_source_attr_map(net.links)
    oriented_src_attrs = build_oriented_source_attr_map(net.links)
    osm_to_new = {nid: NODE_ID_START + i for i, nid in enumerate(graph.nodes)}
    node_rows = []
    for nid, data in graph.nodes(data=True):
        geom = data.get("geometry") or Point(data["x"], data["y"])
        node_rows.append({"node_id": osm_to_new[nid], "geometry": geom})

    node_xy = {nid: (d["x"], d["y"]) for nid, d in graph.nodes(data=True)}
    df = pd.DataFrame(list(graph.edges(data=True)), columns=["_u", "_v", "_data"])
    if len(df) == 0:
        raise ImporterError("OSMnx simplification produced zero links")
    df = df.join(pd.json_normalize(df["_data"]))

    df["a_node"] = df["_u"].map(osm_to_new).astype(np.int64)
    df["b_node"] = df["_v"].map(osm_to_new).astype(np.int64)

    def _resolve_geom(row):
        g = row.get("geometry")
        if is_missing(g) or g.is_empty:
            g = LineString([node_xy[row["_u"]], node_xy[row["_v"]]])
        if isinstance(g, MultiLineString):
            g = max(g.geoms, key=lambda p: p.length)
        return g

    df["geometry"] = df.apply(_resolve_geom, axis=1)

    df["_source_refs"] = _normalize_source_refs(df)
    df = _merge_reciprocal_edges(df).reset_index(drop=True)

    df["link_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["_source_ids"] = df["_source_refs"].apply(lambda refs: _base_source_ids(refs, oriented_src_attrs))
    df[SOURCE_ID_COL] = [
        ids[0] if ids else str(lid) for ids, lid in zip(df["_source_ids"], df["link_id"], strict=True)
    ]

    edge_modes = df["modes"].apply(_coerce_modes) if "modes" in df.columns else pd.Series("c", index=df.index)
    df["modes"] = [_aggregate_modes(sids, src_attrs, m) for sids, m in zip(df["_source_ids"], edge_modes, strict=True)]

    edge_lt = df["link_type"] if "link_type" in df.columns else pd.Series(dtype=object, index=df.index)
    df["link_type"] = [
        _first_non_missing(
            (src_attrs.get(sid, {}).get("link_type") for sid in sids),
            fallback=edge_val,
            default="unknown",
        )
        for sids, edge_val in zip(df["_source_ids"], edge_lt, strict=True)
    ]

    if "length" in df.columns:
        df["distance"] = df["length"].apply(
            lambda v: float(sum(v)) if isinstance(v, list) else float(v) if pd.notna(v) else np.nan
        )
    else:
        df["distance"] = np.nan
    need_dist = df["distance"].isna()
    if need_dist.any():
        gs = gpd.GeoSeries(df.loc[need_dist, "geometry"].values, crs="EPSG:4326")
        df.loc[need_dist, "distance"] = compute_lengths(gs).to_numpy()

    edge_name = df["name"] if "name" in df.columns else pd.Series(dtype=object, index=df.index)
    df["name"] = [
        _first_non_missing((src_attrs.get(sid, {}).get("name") for sid in sids), fallback=edge_val)
        for sids, edge_val in zip(df["_source_ids"], edge_name, strict=True)
    ]

    directional_attrs = [
        _aggregate_directional_attrs(geom, refs, oriented_src_attrs)
        for geom, refs in zip(df["geometry"], df["_source_refs"], strict=True)
    ]
    df["direction"] = [attrs["direction"] for attrs in directional_attrs]
    df["speed_ab"] = [attrs["speed_ab"] for attrs in directional_attrs]
    df["speed_ba"] = [attrs["speed_ba"] for attrs in directional_attrs]
    df["lanes_ab"] = [attrs["lanes_ab"] for attrs in directional_attrs]
    df["lanes_ba"] = [attrs["lanes_ba"] for attrs in directional_attrs]

    df[PROVENANCE_OUT_COL] = df["_source_ids"].apply(lambda ids: build_provenance(ids, src_attrs))

    out_cols = [
        "link_id",
        "a_node",
        "b_node",
        "direction",
        "modes",
        "link_type",
        "distance",
        "geometry",
        "name",
        "speed_ab",
        "speed_ba",
        "lanes_ab",
        "lanes_ba",
        SOURCE_ID_COL,
        PROVENANCE_OUT_COL,
    ]
    links_out = gpd.GeoDataFrame(df[[c for c in out_cols if c in df.columns]], geometry="geometry", crs="EPSG:4326")

    nodes_out = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
    used = set(links_out["a_node"]) | set(links_out["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used)].reset_index(drop=True)
    nodes_out["modes"] = compute_node_modes(nodes_out["node_id"].to_numpy(), links_out, fallback="c")

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


def _merge_reciprocal_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Recombine the two halves of a bidirectional link into a single row.

    ``StagedNetwork.to_graph`` decomposes every two-way staged link into a pair of
    opposing directed edges (tagged ``<base>::ab`` / ``<base>::ba``) because that
    is the topology OSMnx expects. Those halves are simplified independently, so
    without this pass each two-way street returns as *two* one-way links --
    doubling both the link count and the total network length.

    Two edges are recombined when they run between the same node pair in opposite
    directions **and** share at least one base source id, so genuinely distinct
    parallel links (e.g. separate carriageways) are never collapsed. The surviving
    row keeps its own geometry and endpoints and inherits the mate's source refs;
    ``_aggregate_directional_attrs`` then sees both orientations and recovers
    ``direction=0`` plus the per-direction speed/lane values.
    """
    a_nodes = df["a_node"].tolist()
    b_nodes = df["b_node"].tolist()
    refs_col = [list(refs) for refs in df["_source_refs"]]
    bases = [{str(ref).partition("::")[0] for ref in refs} for refs in refs_col]

    pending: dict = {}
    dropped: set = set()

    for pos in range(len(df)):
        mate = None
        bucket = pending.get((b_nodes[pos], a_nodes[pos]))
        if bucket:
            for candidate in bucket:
                if bases[pos] & bases[candidate]:
                    mate = candidate
                    break
        if mate is None:
            pending.setdefault((a_nodes[pos], b_nodes[pos]), []).append(pos)
            continue
        bucket.remove(mate)
        refs_col[mate] = refs_col[mate] + refs_col[pos]
        dropped.add(pos)

    if not dropped:
        return df

    out = df.copy()
    out["_source_refs"] = refs_col
    return out.iloc[[pos for pos in range(len(df)) if pos not in dropped]]


def _normalize_source_refs(df: pd.DataFrame) -> pd.Series:
    if _SOURCE_REF_COL in df.columns:
        return df[_SOURCE_REF_COL].apply(_as_str_list)
    if SOURCE_ID_COL in df.columns:
        return df[SOURCE_ID_COL].apply(
            lambda values: [f"{value}::ab" for value in _as_str_list(values)]
        )
    return pd.Series([[] for _ in range(len(df))], index=df.index)


def _as_str_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _base_source_ids(source_refs: list[str], oriented_src_attrs: dict) -> list[str]:
    source_ids = []
    for source_ref in source_refs:
        source_id = oriented_src_attrs.get(source_ref, {}).get("source_id")
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def _aggregate_modes(source_ids: list, src_attrs: dict, fallback) -> str:
    chars = set()
    for sid in source_ids:
        modes = src_attrs.get(sid, {}).get("modes")
        if isinstance(modes, str):
            chars.update(modes)
    if not chars and isinstance(fallback, str):
        chars.update(fallback)
    return "".join(sorted(chars)) or "c"


def _aggregate_directional_attrs(geom, source_refs: list[str], oriented_src_attrs: dict) -> dict:
    forward = []
    backward = []
    for source_ref in source_refs:
        attrs = oriented_src_attrs.get(source_ref)
        if attrs is None or attrs["geometry"] is None:
            continue
        candidate = forward if aligned_along_geometry(geom, attrs["geometry"]) else backward
        candidate.append((geom.distance(attrs["geometry"]), attrs))

    if forward and backward:
        direction = 0
    elif forward:
        direction = 1
    elif backward:
        direction = -1
    else:
        direction = 0

    return {
        "direction": direction,
        "speed_ab": _nearest_value(forward, "speed"),
        "speed_ba": _nearest_value(backward, "speed"),
        "lanes_ab": _nearest_value(forward, "lanes"),
        "lanes_ba": _nearest_value(backward, "lanes"),
    }


def _nearest_value(candidates: list[tuple[float, dict]], field: str):
    for _distance, attrs in sorted(candidates, key=lambda item: item[0]):
        value = attrs.get(field)
        if not is_missing(value):
            return value
    return None


def _first_non_missing(values, *, fallback=None, default=None):
    for value in values:
        if not is_missing(value):
            return value
    if not is_missing(fallback):
        return fallback
    return default
