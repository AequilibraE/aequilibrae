import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from shapely.geometry import Point
from typing import Sequence

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import is_missing
from aequilibrae.project.network.importer.schema.modes import filter_by_modes, requested_mode_codes
from aequilibrae.project.network.importer.sources.osm.tags_to_ir import (
    directional_lanes,
    directional_speeds,
    modes_for_tags,
    normalise_tag_key,
    parse_direction,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.importer.utils import NODE_ID_START, compute_lengths, compute_node_modes
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)

_RESERVED_LINK_COLS = {
    "a_node",
    "b_node",
    "link_id",
    "distance",
    "modes",
    "direction",
    "speed_ab",
    "speed_ba",
    "lanes_ab",
    "lanes_ba",
    "link_type",
    "name",
    "geometry",
    "source_id",
}
_NON_TAG_COLS = {"u", "v", "key", "a_node", "b_node", "geometry", "distance", "osmid", "osm_id"}

def acquire_overpass(
    *,
    modes: Sequence[str],
    download_cache: DownloadCache,
    model_area=None,
    place_name=None,
    custom_filter=None,
) -> StagedNetwork:
    ox = require("osmnx", feature="OSM Overpass download")
    _configure_osmnx(ox)

    if (model_area is None) == (place_name is None):
        raise ImporterError("The osm-overpass source requires exactly one of `model_area` or `place_name`")

    # An explicit area is the caller's choice, so it is never substituted for another one.
    if model_area is not None:
        return _fetch_and_stage(
            ox, modes=modes, download_cache=download_cache, model_area=model_area, custom_filter=custom_filter
        )

    return _fetch_and_stage(
        ox, modes=modes, download_cache=download_cache, place_name=place_name, custom_filter=custom_filter
    )


def _fetch_and_stage(ox, *, modes, download_cache, custom_filter, model_area=None, place_name=None) -> StagedNetwork:
    source_url = (
        f"overpass:bbox={list(model_area.bounds)}" if model_area is not None else f"overpass:place={place_name}"
    )

    from osmnx._errors import InsufficientResponseError

    fetch_kwargs = {"network_type": "all", "simplify": False, "retain_all": True, "custom_filter": custom_filter}
    try:
        if model_area is not None:
            graph = ox.graph_from_polygon(model_area, **fetch_kwargs)
        else:
            graph = ox.graph_from_place(place_name, **fetch_kwargs)
    except InsufficientResponseError as exc:
        raise ImporterError(f"Overpass returned an empty or partial response for {source_url}: {exc}") from exc
    except Exception as exc:
        from requests.exceptions import RequestException

        if isinstance(exc, RequestException):
            raise ImporterError(f"Overpass request failed: {exc}") from exc
        raise exc

    if graph is None or graph.number_of_edges() == 0:
        raise ImporterError(f"Overpass returned no edges for the requested area ({source_url})")

    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes_gdf = nodes_gdf.reset_index()
    edges_gdf = _collapse_reciprocal_edges(edges_gdf.reset_index())

    _persist_overpass_payload(
        download_cache=download_cache,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        place_name=place_name,
        model_area=model_area,
        modes=modes,
        custom_filter=custom_filter,
    )

    source_meta = {
        "source": "osm",
        "backend": "osmnx-overpass",
        "source_url": source_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    net = _edges_nodes_to_staged(edges_gdf, nodes_gdf, modes=modes, source_meta=source_meta, clip_to=model_area)
    # Validated here as well as in NetworkImporter.run() so that a bad response is caught
    # while there is still another query area left to try.
    net.validate()
    return net


def _collapse_reciprocal_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop the reversed half of each reciprocal OSMnx edge pair.

    Keeps reversed-only rows (``oneway=-1``) that have no forward counterpart.
    """
    if "reversed" not in edges.columns or len(edges) == 0:
        return edges

    def _flag(value):
        # OSMnx may carry a list here when parallel edges were merged.
        if isinstance(value, (list, tuple, set)):
            return bool(next(iter(value))) if value else False
        return bool(value)

    reversed_flags = edges["reversed"].map(_flag).to_numpy()
    way_ids = edges["osmid"] if "osmid" in edges.columns else edges.index
    keys = [(frozenset((u, v)), str(w)) for u, v, w in zip(edges["u"], edges["v"], way_ids, strict=True)]
    forward_keys = {key for key, is_reversed in zip(keys, reversed_flags, strict=True) if not is_reversed}

    keep, kept_keys = [], set()
    for key, is_reversed in zip(keys, reversed_flags, strict=True):
        # Drop a reversed row only when its forward twin is present.
        if is_reversed and key in forward_keys:
            keep.append(False)
        elif key in kept_keys:
            keep.append(False)
        else:
            kept_keys.add(key)
            keep.append(True)

    dropped = len(edges) - sum(keep)
    if dropped:
        logger.info(f"Collapsed {dropped} reciprocal OSMnx edges into their forward counterparts")
    return edges[keep].reset_index(drop=True)


def _configure_osmnx(ox) -> None:
    from aequilibrae.parameters import Parameters

    params = Parameters().parameters.get("osm", {}) or {}
    if "overpass_endpoint" in params:
        # osmnx appends "/interpreter" itself, so this must stay a base URL
        # (e.g. "https://overpass-api.de/api"). Appending it here too yields
        # ".../interpreter/interpreter" and the server rejects every request.
        ox.settings.overpass_url = params["overpass_endpoint"].rstrip("/")
    if "nominatim_endpoint" in params:
        ox.settings.nominatim_url = params["nominatim_endpoint"]
    if "timeout" in params:
        # osmnx 2.x calls this ``requests_timeout``; assigning to ``timeout``
        # silently creates an attribute the library never reads.
        ox.settings.requests_timeout = int(params["timeout"])
    if "overpass_rate_limit" in params:
        ox.settings.overpass_rate_limit = bool(params["overpass_rate_limit"])
    if "accept_language" in params:
        ox.settings.http_accept_language = params["accept_language"]


def _persist_overpass_payload(
    *,
    download_cache: DownloadCache,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    place_name,
    model_area,
    modes,
    custom_filter,
) -> None:
    combined = pd.concat(
        [nodes_gdf.assign(feature_type="node"), edges_gdf.assign(feature_type="edge")],
        ignore_index=True,
    )
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=nodes_gdf.crs or "EPSG:4326")
    download_cache.write_geoparquet("osm.parquet", combined_gdf)
    download_cache.write_manifest(
        {
            "source": "osm-overpass",
            "backend": "osmnx",
            "place_name": place_name,
            "bbox": list(model_area.bounds) if model_area is not None else None,
            "modes": list(modes),
            "custom_filter": custom_filter,
            "n_nodes": int(len(nodes_gdf)),
            "n_edges": int(len(edges_gdf)),
        }
    )


def acquire_pbf(
    *,
    pbf_path: Path,
    modes: Sequence[str],
    download_cache: DownloadCache,
) -> StagedNetwork:
    pyrosm = require("pyrosm", feature="OSM PBF reading")

    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    osm = pyrosm.OSM(str(pbf_path))
    result = osm.get_network(network_type="all", nodes=True)
    if result is None:
        raise ImporterError(f"pyrosm returned no data from {pbf_path}")
    nodes_raw, edges = result
    if edges is None or len(edges) == 0:
        raise ImporterError(f"pyrosm returned no edges from {pbf_path}")

    nodes = _pyrosm_nodes_frame(nodes_raw, edges)

    source_meta = {
        "source": "osm",
        "backend": "pyrosm",
        "source_url": str(pbf_path),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return _edges_nodes_to_staged(edges, nodes, modes=modes, source_meta=source_meta, clip_to=None)


def _first_last_points(geom):
    if geom is None or geom.is_empty:
        return None, None
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        return (Point(coords[0]), Point(coords[-1])) if coords else (None, None)
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
        if not parts:
            return None, None
        first_coords = list(parts[0].coords)
        last_coords = list(parts[-1].coords)
        if not first_coords or not last_coords:
            return None, None
        return Point(first_coords[0]), Point(last_coords[-1])
    return None, None


def _pyrosm_nodes_frame(nodes_raw: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    used_ids: set = set()
    for col in ("u", "v"):
        if col in edges.columns:
            used_ids.update(edges[col].dropna().astype("int64").tolist())
    if not used_ids:
        raise ImporterError("pyrosm edges frame has no 'u'/'v' OSM node ids")

    keep = (
        nodes_raw.assign(id=nodes_raw["id"].astype("int64"))
        .loc[lambda df: df["id"].isin(used_ids), ["id", "geometry"]]
        .rename(columns={"id": "osm_id"})
    )

    missing_ids = used_ids - set(keep["osm_id"].tolist())
    if missing_ids:
        candidates = edges.loc[edges["u"].isin(missing_ids) | edges["v"].isin(missing_ids)]
        synth: dict = {}
        for row in candidates.itertuples(index=False):
            first, last = _first_last_points(row.geometry)
            if first is None:
                continue
            u = int(row.u) if pd.notna(row.u) else None
            v = int(row.v) if pd.notna(row.v) else None
            if u in missing_ids and u not in synth:
                synth[u] = first
            if v in missing_ids and v not in synth:
                synth[v] = last
        if synth:
            extra = gpd.GeoDataFrame(
                {"osm_id": list(synth), "geometry": list(synth.values())},
                geometry="geometry",
                crs=keep.crs or "EPSG:4326",
            )
            keep = pd.concat([keep, extra], ignore_index=True)

    return gpd.GeoDataFrame(keep, geometry="geometry", crs="EPSG:4326")


def _edges_nodes_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    *,
    modes: Sequence[str],
    source_meta: dict,
    clip_to,
) -> StagedNetwork:
    requested_codes = requested_mode_codes(modes)
    nodes_gdf, osm_to_node = _prepare_nodes(nodes_gdf)
    edges = _prepare_edges(edges_gdf, osm_to_node)
    edges = _add_osm_attributes(edges, requested_codes)
    edges = _filter_edges(edges, modes, clip_to)
    edges = _finalize_edges(edges)
    nodes_out = _staged_nodes(nodes_gdf, edges)
    links_out = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326")
    return StagedNetwork(nodes=nodes_out, links=links_out, source_meta=source_meta)


def _prepare_nodes(nodes_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    if "osm_id" not in nodes_gdf.columns and "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"osmid": "osm_id"})
    if "osm_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.reset_index()
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    nodes_gdf["osm_id"] = nodes_gdf["osm_id"].astype("int64")
    nodes_gdf = nodes_gdf.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
    nodes_gdf["node_id"] = np.arange(
        NODE_ID_START,
        NODE_ID_START + len(nodes_gdf),
        dtype=np.int64,
    )
    return nodes_gdf, dict(zip(nodes_gdf["osm_id"], nodes_gdf["node_id"], strict=True))


def _prepare_edges(edges_gdf: gpd.GeoDataFrame, osm_to_node: dict) -> gpd.GeoDataFrame:
    if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
        raise ImporterError("Edges frame must contain 'u' and 'v' columns (OSM node ids)")
    edges = edges_gdf.to_crs("EPSG:4326")
    before = len(edges)
    edges = edges[edges["u"].isin(osm_to_node) & edges["v"].isin(osm_to_node)].copy()
    dropped_unmapped = before - len(edges)
    edges["a_node"] = edges["u"].map(osm_to_node).astype("int64")
    edges["b_node"] = edges["v"].map(osm_to_node).astype("int64")
    before_geometry = len(edges)
    edges = edges[~edges.geometry.isna()].reset_index(drop=True)
    dropped_geometry = before_geometry - len(edges)
    if dropped_unmapped or dropped_geometry:
        logger.info(f"Dropped {dropped_unmapped} OSM edges with unmapped nodes and {dropped_geometry} without geometry")
    if len(edges) == 0:
        raise ImporterError("OSM acquisition produced zero usable edges after node mapping")
    edges["distance"] = compute_lengths(edges.geometry).to_numpy()
    return edges


def _add_osm_attributes(edges: gpd.GeoDataFrame, requested_codes: set) -> gpd.GeoDataFrame:
    records = edges[[c for c in edges.columns if c not in _NON_TAG_COLS]].to_dict(orient="records")
    modes_strs, directions = [], []
    speed_abs, speed_bas = [], []
    lanes_abs, lanes_bas = [], []
    link_types, names = [], []

    for tags in records:
        tags = {k: v for k, v in tags.items() if not is_missing(v)}
        highway = tags.get("highway")
        if isinstance(highway, list) and highway:
            tags["highway"] = highway[0]
        modes_strs.append(filter_by_modes(modes_for_tags(tags), requested_codes))
        directions.append(parse_direction(tags))
        speed_ab, speed_ba = directional_speeds(tags)
        speed_abs.append(speed_ab)
        speed_bas.append(speed_ba)
        lanes_ab, lanes_ba = directional_lanes(tags)
        lanes_abs.append(lanes_ab)
        lanes_bas.append(lanes_ba)
        link_types.append(str(tags.get("highway") or "unknown"))
        names.append(tags.get("name"))

    return edges.assign(
        modes=modes_strs,
        direction=directions,
        speed_ab=speed_abs,
        speed_ba=speed_bas,
        lanes_ab=lanes_abs,
        lanes_ba=lanes_bas,
        link_type=link_types,
        name=names,
    )


def _filter_edges(edges: gpd.GeoDataFrame, modes: Sequence[str], clip_to) -> gpd.GeoDataFrame:
    before = len(edges)
    edges = edges[edges["modes"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Mode filter kept {len(edges)} / {before} links")
    if len(edges) == 0:
        raise ImporterError(f"After mode filtering ({modes!r}) no links remain")

    if clip_to is not None:
        before = len(edges)
        edges = edges[edges.geometry.intersects(clip_to)].reset_index(drop=True)
        logger.info(f"Model-area clip kept {len(edges)} / {before} links")
        if len(edges) == 0:
            raise ImporterError("Model-area clip removed all OSM links")
    return edges


def _finalize_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    if "osmid" in edges.columns:
        edges["source_id"] = edges["osmid"].map(lambda v: str(v[0]) if isinstance(v, list) and v else str(v))
    elif "osm_id" in edges.columns:
        edges["source_id"] = edges["osm_id"].astype(str)
    elif "id" in edges.columns:
        edges["source_id"] = edges["id"].astype(str)
    else:
        edges["source_id"] = edges["link_id"].astype(str)

    edges = edges.drop(columns=[c for c in ("u", "v", "key", "osmid") if c in edges.columns])
    rename_map = {
        c: normalise_tag_key(c) for c in edges.columns if c not in _RESERVED_LINK_COLS and normalise_tag_key(c) != c
    }
    return edges.rename(columns=rename_map) if rename_map else edges


def _staged_nodes(nodes_gdf: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    used_nodes = set(edges["a_node"]) | set(edges["b_node"])
    nodes_out = gpd.GeoDataFrame(
        {
            "node_id": nodes_gdf["node_id"].astype(np.int64),
            "geometry": nodes_gdf["geometry"],
            "modes": compute_node_modes(nodes_gdf["node_id"].to_numpy(), edges, fallback="c"),
            "source_id": nodes_gdf["osm_id"].astype(str),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return nodes_out[nodes_out["node_id"].isin(used_nodes)].reset_index(drop=True)
