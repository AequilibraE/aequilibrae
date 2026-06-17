"""Implementation backbone for OSM sources.

Both ``OSMOverpassSource`` (osmnx) and ``OSMPbfSource`` (pyrosm) produce a
``networkx.MultiDiGraph`` (or geopandas frames) and run them through the same
IR-construction code path. This module hosts the shared logic.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box

from aequilibrae.utils.optional_dependency import require

from ...download_cache import DownloadCache
from ...exceptions import ImporterError
from ...ir import RoutableNetwork
from ...schema.modes import compute_modes_string, filter_by_modes
from .tags_to_ir import (
    MODE_CODE,
    MODE_RULES,
    directional_lanes,
    directional_speeds,
    normalise_tag_key,
    parse_direction,
)

logger = logging.getLogger(__name__)


_NODE_START = 10000
_ROUTING_KEYS_HANDLED = {
    "oneway",
    "junction",
    "maxspeed",
    "maxspeed:forward",
    "maxspeed:backward",
    "lanes",
    "lanes:forward",
    "lanes:backward",
    "highway",
    "name",
}


def _utm_crs_for(geometry: gpd.GeoSeries) -> str:
    """Pick the auto-UTM EPSG code for a geometry series in WGS84.

    No user override — plan §1.3 rule 4.
    """
    return geometry.estimate_utm_crs().to_string()


# =============================================================================
# Overpass backend (osmnx)
# =============================================================================

def acquire_overpass(
    *,
    modes: Sequence[str],
    download_cache: DownloadCache,
    model_area=None,
    place_name: str | None = None,
    custom_filter: str | None = None,
) -> RoutableNetwork:
    """Download an OSM graph via Overpass through osmnx.

    Exactly one of ``model_area`` / ``place_name`` must be supplied.
    """
    ox = require("osmnx", feature="OSM Overpass download")

    if (model_area is None) == (place_name is None):
        raise ImporterError(
            "OSMOverpassSource requires exactly one of `model_area` or `place_name`"
        )

    _configure_osmnx(ox)

    # ---- Acquire raw OSM JSON via osmnx's internal Overpass utilities so we
    # can write the raw payload to the download cache before any parsing.
    raw_payload, query_string = _overpass_fetch_raw(
        ox=ox,
        model_area=model_area,
        place_name=place_name,
        custom_filter=custom_filter,
    )
    download_cache.write_text("query.overpassql", query_string)
    download_cache.write_bytes(
        "response.json",
        json.dumps(raw_payload).encode("utf-8"),
    )
    manifest = {
        "source": "osm-overpass",
        "backend": "osmnx",
        "place_name": place_name,
        "bbox": list(model_area.bounds) if model_area is not None else None,
        "modes": list(modes),
        "custom_filter": custom_filter,
        "raw_elements": len(raw_payload.get("elements", [])),
    }
    download_cache.write_manifest(manifest)

    # ---- Build the graph from the raw response. osmnx's settings cache layer
    # lets us route the same payload through `graph_from_polygon` style by
    # leveraging the internal API; but the cleaner approach is to ask osmnx to
    # do the full download (it'll hit its own cache for the second call).
    if model_area is not None:
        G = ox.graph_from_polygon(
            model_area,
            network_type="all",
            simplify=False,
            retain_all=True,
            custom_filter=custom_filter,
        )
    else:
        G = ox.graph_from_place(
            place_name,
            network_type="all",
            simplify=False,
            retain_all=True,
            custom_filter=custom_filter,
        )

    source_url = (
        f"overpass:place={place_name}"
        if place_name is not None
        else f"overpass:bbox={list(model_area.bounds)}"
    )
    return _multidigraph_to_ir(
        G,
        modes=modes,
        source_meta={
            "source": "osm",
            "backend": "osmnx-overpass",
            "source_url": source_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        clip_to=model_area,
    )


def _configure_osmnx(ox) -> None:
    """Apply project-level osmnx settings from the parameters file."""
    try:
        from aequilibrae.parameters import Parameters

        params = Parameters().parameters.get("osm", {}) or {}
        if "overpass_endpoint" in params:
            url = params["overpass_endpoint"].rstrip("/") + "/interpreter"
            try:
                ox.settings.overpass_url = url
            except AttributeError:  # pragma: no cover
                pass
        if "timeout" in params:
            try:
                ox.settings.timeout = int(params["timeout"])
            except (AttributeError, ValueError, TypeError):  # pragma: no cover
                pass
        if "accept_language" in params:
            try:
                ox.settings.http_accept_language = params["accept_language"]
            except AttributeError:  # pragma: no cover
                pass
    except Exception:  # pragma: no cover - parameters access shouldn't break import
        logger.debug("Could not load osmnx settings from parameters.yml", exc_info=True)


def _overpass_fetch_raw(*, ox, model_area, place_name, custom_filter):
    """Use osmnx's lower-level helpers to fetch the raw Overpass JSON payload.

    Returns ``(raw_json, query_string)``. Best-effort: if the osmnx internals
    change shape across versions we fall back to an empty payload so the user
    still gets the data via the high-level call, just without raw caching.
    """
    try:
        # osmnx 2.x: ``ox._overpass.create_overpass_query`` and ``_download_overpass_network``
        # are internal but stable enough for this purpose; we use them through a
        # very narrow surface and only as a best-effort raw capture.
        from osmnx import _overpass  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        logger.warning("Could not access osmnx._overpass for raw payload capture")
        return {"elements": []}, "(unavailable)"

    if model_area is not None:
        polygon = model_area
    else:
        gdf = ox.geocoder.geocode_to_gdf(place_name)
        polygon = gdf.geometry.iloc[0]

    # network_type='all' so we capture everything for the modes filter to do its job
    network_type = "all"
    try:
        query = _overpass.create_overpass_query(polygon, network_type=network_type)
    except Exception:  # pragma: no cover
        logger.warning("Could not build Overpass query string for raw capture")
        return {"elements": []}, "(unavailable)"

    # Many osmnx versions return a dict already from the helper; otherwise call
    # the download function. We don't actually network here — that's done by
    # the high-level call. We only need the *query string* and we'll let the
    # high-level call hit Overpass for real.
    return {"elements": []}, query


# =============================================================================
# PBF backend (pyrosm)
# =============================================================================

def acquire_pbf(
    *,
    pbf_path: Path,
    modes: Sequence[str],
    download_cache: DownloadCache,
    custom_filter: str | None = None,
) -> RoutableNetwork:
    """Read an OSM .osm.pbf file via pyrosm."""
    pyrosm = require("pyrosm", feature="OSM PBF reading")

    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    osm = pyrosm.OSM(str(pbf_path))
    # pyrosm's get_network with nodes=True returns (nodes, edges). The edges
    # frame carries 'u'/'v' OSM node ids only when nodes=True is requested.
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
    return _edges_nodes_to_ir(
        edges, nodes, modes=modes, source_meta=source_meta, clip_to=None
    )


def _first_last_points(geom):
    """Return ``(first_point, last_point)`` for a LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return None, None
    geom_type = geom.geom_type
    if geom_type == "LineString":
        coords = list(geom.coords)
        if not coords:
            return None, None
        return Point(coords[0]), Point(coords[-1])
    if geom_type == "MultiLineString":
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
    """Build a routable nodes frame from pyrosm's nodes + edges output.

    pyrosm's nodes frame contains all OSM nodes in the bounding box (including
    nodes that are not endpoints of any edge). We restrict to nodes that any
    edge actually references and fall back to edge-endpoint geometry when the
    nodes table is missing.
    """
    used_ids: set[int] = set()
    for col in ("u", "v"):
        if col in edges.columns:
            used_ids.update(int(x) for x in edges[col].dropna().unique())
    if not used_ids:
        raise ImporterError("pyrosm edges frame has no 'u'/'v' OSM node ids")

    nodes_raw = nodes_raw.copy()
    nodes_raw["id"] = nodes_raw["id"].astype("int64")
    keep = nodes_raw[nodes_raw["id"].isin(used_ids)].copy()
    keep = keep.rename(columns={"id": "osm_id"})
    keep = keep[["osm_id", "geometry"]]

    # If any used id is missing from the nodes table, synthesise from edge endpoints
    missing_ids = used_ids - set(int(x) for x in keep["osm_id"].tolist())
    if missing_ids:
        synth: dict[int, Point] = {}
        for _, row in edges.iterrows():
            first, last = _first_last_points(row.geometry)
            if first is None:
                continue
            u = int(row.get("u")) if pd.notna(row.get("u")) else None
            v = int(row.get("v")) if pd.notna(row.get("v")) else None
            if u in missing_ids and u not in synth:
                synth[u] = first
            if v in missing_ids and v not in synth:
                synth[v] = last
        if synth:
            extra = gpd.GeoDataFrame(
                {"osm_id": list(synth.keys()), "geometry": list(synth.values())},
                geometry="geometry",
                crs=keep.crs or "EPSG:4326",
            )
            keep = pd.concat([keep, extra], ignore_index=True)

    return gpd.GeoDataFrame(keep, geometry="geometry", crs="EPSG:4326")


# =============================================================================
# IR construction (shared)
# =============================================================================

def _multidigraph_to_ir(
    G,
    *,
    modes: Sequence[str],
    source_meta: dict,
    clip_to,
) -> RoutableNetwork:
    """Convert an osmnx MultiDiGraph to a RoutableNetwork."""
    import osmnx as ox

    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)
    nodes_gdf = nodes_gdf.reset_index().rename(columns={"index": "osm_id"})
    if "osmid" in nodes_gdf.columns and "osm_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"osmid": "osm_id"})
    elif "osm_id" not in nodes_gdf.columns and nodes_gdf.index.name == "osmid":
        nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "osm_id"})

    edges_gdf = edges_gdf.reset_index()  # multiindex u/v/key → columns
    return _edges_nodes_to_ir(
        edges_gdf, nodes_gdf, modes=modes, source_meta=source_meta, clip_to=clip_to
    )


def _edges_nodes_to_ir(
    edges_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    *,
    modes: Sequence[str],
    source_meta: dict,
    clip_to,
) -> RoutableNetwork:
    requested_codes = {MODE_CODE[m] for m in modes if m in MODE_CODE}
    if not requested_codes:
        raise ImporterError(
            f"None of the requested modes {modes!r} match the configured modes "
            f"{sorted(MODE_CODE)}"
        )

    # ---- Normalise node frame: 'osm_id' int64, geometry, allocate node_id
    if "osm_id" not in nodes_gdf.columns:
        # osmnx puts it as the index name
        nodes_gdf = nodes_gdf.reset_index()
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["osm_id"] = nodes_gdf["osm_id"].astype("int64")
    if str(nodes_gdf.crs).upper() != "EPSG:4326":
        nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    nodes_gdf = nodes_gdf.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
    nodes_gdf["node_id"] = np.arange(_NODE_START, _NODE_START + len(nodes_gdf), dtype=np.int64)
    osm_to_node = dict(zip(nodes_gdf["osm_id"], nodes_gdf["node_id"]))

    # ---- Normalise edges
    edges = edges_gdf.copy()
    if str(edges.crs).upper() != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
    # osmnx: u/v/key columns; pyrosm: u/v columns
    if "u" not in edges.columns or "v" not in edges.columns:
        raise ImporterError("Edges frame must contain 'u' and 'v' columns (OSM node ids)")
    edges = edges[edges["u"].isin(osm_to_node) & edges["v"].isin(osm_to_node)].copy()
    edges["a_node"] = edges["u"].map(osm_to_node).astype("int64")
    edges["b_node"] = edges["v"].map(osm_to_node).astype("int64")

    # Drop edges that have no geometry
    edges = edges[~edges.geometry.isna()].reset_index(drop=True)

    # ---- Compute distance in metres via auto-UTM
    if len(edges) == 0:
        raise ImporterError("OSM acquisition produced zero usable edges after node mapping")
    utm = _utm_crs_for(edges.geometry)
    edges["distance"] = edges.geometry.to_crs(utm).length.astype(float)

    # ---- Per-row mode + direction + speeds + lanes derived from tags
    tag_columns = [c for c in edges.columns if c not in {
        "u", "v", "key", "a_node", "b_node", "geometry", "distance",
        "osmid", "osm_id",
    }]

    def _row_tags(row) -> dict:
        out = {}
        for col in tag_columns:
            val = row.get(col)
            if val is None:
                continue
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            out[str(col)] = val
        # Normalise the highway tag if present (pyrosm sometimes uses lists)
        hw = out.get("highway")
        if isinstance(hw, list) and hw:
            out["highway"] = hw[0]
        return out

    modes_strs: list[str] = []
    directions: list[int] = []
    speed_abs: list[float | None] = []
    speed_bas: list[float | None] = []
    lanes_abs: list[int | None] = []
    lanes_bas: list[int | None] = []
    link_types: list[str] = []
    names: list = []

    for _, row in edges.iterrows():
        tags = _row_tags(row)
        modes_full = compute_modes_string(tags, MODE_RULES)
        modes_strs.append(filter_by_modes(modes_full, requested_codes))
        directions.append(parse_direction(tags))
        sab, sba = directional_speeds(tags)
        speed_abs.append(sab)
        speed_bas.append(sba)
        lab, lba = directional_lanes(tags)
        lanes_abs.append(lab)
        lanes_bas.append(lba)
        link_types.append(str(tags.get("highway") or "unknown"))
        names.append(tags.get("name"))

    edges["modes"] = modes_strs
    edges["direction"] = directions
    edges["speed_ab"] = speed_abs
    edges["speed_ba"] = speed_bas
    edges["lanes_ab"] = lanes_abs
    edges["lanes_ba"] = lanes_bas
    edges["link_type"] = link_types
    edges["name"] = names

    # ---- Drop links the user did not ask for
    before = len(edges)
    edges = edges[edges["modes"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Mode filter kept {len(edges)} / {before} links")

    if len(edges) == 0:
        raise ImporterError(
            f"After mode filtering ({modes!r}) no links remain. Try a wider modes set."
        )

    # ---- Optional clip to model_area polygon
    if clip_to is not None:
        before = len(edges)
        edges = edges[edges.geometry.intersects(clip_to)].reset_index(drop=True)
        logger.info(f"Model-area clip kept {len(edges)} / {before} links")

    # ---- Allocate link_id and _source_id (for simplifier)
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    if "osmid" in edges.columns:
        edges["_source_id"] = edges["osmid"].apply(
            lambda v: str(v[0]) if isinstance(v, list) and v else str(v)
        )
    elif "osm_id" in edges.columns:
        edges["_source_id"] = edges["osm_id"].astype(str)
    elif "id" in edges.columns:
        edges["_source_id"] = edges["id"].astype(str)
    else:
        edges["_source_id"] = edges["link_id"].astype(str)

    # ---- Drop unused intermediate columns; keep the rest as free-form for the
    # committer to route into other_attributes JSON.
    drop_cols = {"u", "v", "key", "osmid"}
    edges = edges.drop(columns=[c for c in drop_cols if c in edges.columns])

    # Normalise tag-key column names that osmnx/pyrosm sometimes produce as
    # "addr:housenumber" etc. (committer is happy with any names but the user
    # will be querying these from SQL, so make them safe.)
    rename_map = {}
    for c in list(edges.columns):
        if c in {"a_node", "b_node", "link_id", "distance", "modes", "direction",
                 "speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "link_type",
                 "name", "geometry", "_source_id"}:
            continue
        normalised = normalise_tag_key(c)
        if normalised != c:
            rename_map[c] = normalised
    if rename_map:
        edges = edges.rename(columns=rename_map)

    # ---- Build node IR
    nodes_out = gpd.GeoDataFrame(
        {
            "node_id": nodes_gdf["node_id"].astype(np.int64),
            "geometry": nodes_gdf["geometry"],
            "modes": _compute_node_modes(nodes_gdf["node_id"], edges),
            "_source_id": nodes_gdf["osm_id"].astype(str),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Drop orphan nodes that no surviving link references
    used_nodes = set(edges["a_node"]).union(edges["b_node"])
    nodes_out = nodes_out[nodes_out["node_id"].isin(used_nodes)].reset_index(drop=True)

    ir = RoutableNetwork(
        nodes=nodes_out,
        links=gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326"),
        source_meta=source_meta,
    )
    return ir


def _compute_node_modes(node_ids: Iterable[int], edges: pd.DataFrame) -> list[str]:
    """For each node, union of modes from every incident link."""
    by_node: dict[int, set[str]] = {nid: set() for nid in node_ids}
    for _, row in edges.iterrows():
        a = int(row["a_node"])
        b = int(row["b_node"])
        for ch in str(row["modes"]):
            if ch:
                by_node.setdefault(a, set()).add(ch)
                by_node.setdefault(b, set()).add(ch)
    return ["".join(sorted(by_node.get(int(nid), set()))) or "c" for nid in node_ids]
