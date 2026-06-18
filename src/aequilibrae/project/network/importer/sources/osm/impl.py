"""Implementation backbone for OSM sources.

Both ``OSMOverpassSource`` (osmnx) and ``OSMPbfSource`` (pyrosm) produce
geopandas frames and run them through the same staged-network construction
code path.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.modes import compute_modes_string, filter_by_modes
from aequilibrae.project.network.importer.sources.osm.tags_to_ir import (
    MODE_CODE,
    MODE_RULES,
    directional_lanes,
    directional_speeds,
    normalise_tag_key,
    parse_direction,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)


_NODE_START = 10000
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
    "_source_id",
}
_NON_TAG_COLS = {"u", "v", "key", "a_node", "b_node", "geometry", "distance", "osmid", "osm_id"}


# =============================================================================
# Overpass backend (osmnx)
# =============================================================================


def acquire_overpass(
    *,
    modes: Sequence[str],
    download_cache: DownloadCache,
    model_area=None,
    place_name=None,
    custom_filter=None,
) -> StagedNetwork:
    """Download an OSM graph via Overpass through osmnx.

    Exactly one of ``model_area`` / ``place_name`` must be supplied. Raises
    :class:`ImporterError` on Overpass HTTP errors and on empty responses.
    """
    ox = require("osmnx", feature="OSM Overpass download")
    _configure_osmnx(ox)

    if (model_area is None) == (place_name is None):
        raise ImporterError("OSMOverpassSource requires exactly one of `model_area` or `place_name`")

    source_url = (
        f"overpass:place={place_name}" if place_name is not None else f"overpass:bbox={list(model_area.bounds)}"
    )

    fetch_kwargs = dict(network_type="all", simplify=False, retain_all=True, custom_filter=custom_filter)
    try:
        if model_area is not None:
            G = ox.graph_from_polygon(model_area, **fetch_kwargs)
        else:
            G = ox.graph_from_place(place_name, **fetch_kwargs)
    except ox.exceptions.InsufficientResponseError as exc:
        raise ImporterError(f"Overpass returned an empty or partial response for {source_url}: {exc}") from exc
    except Exception as exc:
        from requests.exceptions import RequestException

        if isinstance(exc, RequestException):
            raise ImporterError(
                f"Overpass request failed: {exc}. Check connectivity and the "
                f"endpoint configured in parameters.yml::osm.overpass_endpoint."
            ) from exc
        raise

    if G is None or G.number_of_edges() == 0:
        raise ImporterError(
            f"Overpass returned no edges for the requested area ({source_url}). Widen the bbox or adjust custom_filter."
        )

    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)
    nodes_gdf = nodes_gdf.reset_index()
    edges_gdf = edges_gdf.reset_index()

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
    return _edges_nodes_to_staged(edges_gdf, nodes_gdf, modes=modes, source_meta=source_meta, clip_to=model_area)


def _configure_osmnx(ox) -> None:
    """Apply project-level osmnx settings from the parameters file."""
    from aequilibrae.parameters import Parameters

    params = Parameters().parameters.get("osm", {}) or {}
    if "overpass_endpoint" in params:
        ox.settings.overpass_url = params["overpass_endpoint"].rstrip("/") + "/interpreter"
    if "timeout" in params:
        ox.settings.timeout = int(params["timeout"])
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
    """Consolidate (nodes, edges) into a single GeoDataFrame and persist as GeoParquet."""
    combined = pd.concat(
        [nodes_gdf.assign(feature_type="node"), edges_gdf.assign(feature_type="edge")],
        ignore_index=True,
    )
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=nodes_gdf.crs or "EPSG:4326")
    download_cache.write_geoparquet("osm.parquet", combined_gdf)
    download_cache.write_manifest({
        "source": "osm-overpass",
        "backend": "osmnx",
        "place_name": place_name,
        "bbox": list(model_area.bounds) if model_area is not None else None,
        "modes": list(modes),
        "custom_filter": custom_filter,
        "n_nodes": int(len(nodes_gdf)),
        "n_edges": int(len(edges_gdf)),
    })


# =============================================================================
# PBF backend (pyrosm)
# =============================================================================


def acquire_pbf(
    *,
    pbf_path: Path,
    modes: Sequence[str],
    download_cache: DownloadCache,
    custom_filter=None,
) -> StagedNetwork:
    """Read an OSM .osm.pbf file via pyrosm."""
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
    """Return ``(first_point, last_point)`` for a LineString or MultiLineString."""
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
    """Build a staged nodes frame from pyrosm's nodes + edges output."""
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
        # Only iterate edges that reference a missing endpoint
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
                geometry="geometry", crs=keep.crs or "EPSG:4326",
            )
            keep = pd.concat([keep, extra], ignore_index=True)

    return gpd.GeoDataFrame(keep, geometry="geometry", crs="EPSG:4326")


# =============================================================================
# Staged-network construction (shared)
# =============================================================================


def _edges_nodes_to_staged(
    edges_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    *,
    modes: Sequence[str],
    source_meta: dict,
    clip_to,
) -> StagedNetwork:
    requested_codes = {MODE_CODE[m] for m in modes if m in MODE_CODE}
    if not requested_codes:
        raise ImporterError(f"None of the requested modes {modes!r} match the configured modes {sorted(MODE_CODE)}")

    # ---- Nodes
    if "osm_id" not in nodes_gdf.columns and "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"osmid": "osm_id"})
    if "osm_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.reset_index()
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    nodes_gdf["osm_id"] = nodes_gdf["osm_id"].astype("int64")
    nodes_gdf = nodes_gdf.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
    nodes_gdf["node_id"] = np.arange(_NODE_START, _NODE_START + len(nodes_gdf), dtype=np.int64)
    osm_to_node = dict(zip(nodes_gdf["osm_id"], nodes_gdf["node_id"]))

    # ---- Edges
    if "u" not in edges_gdf.columns or "v" not in edges_gdf.columns:
        raise ImporterError("Edges frame must contain 'u' and 'v' columns (OSM node ids)")
    edges = edges_gdf.to_crs("EPSG:4326")
    edges = edges[edges["u"].isin(osm_to_node) & edges["v"].isin(osm_to_node)].copy()
    edges["a_node"] = edges["u"].map(osm_to_node).astype("int64")
    edges["b_node"] = edges["v"].map(osm_to_node).astype("int64")
    edges = edges[~edges.geometry.isna()].reset_index(drop=True)

    if len(edges) == 0:
        raise ImporterError("OSM acquisition produced zero usable edges after node mapping")

    # ---- Distance via auto-UTM (vectorised)
    utm = edges.geometry.estimate_utm_crs().to_string()
    edges["distance"] = edges.geometry.to_crs(utm).length.astype(float)

    # ---- Per-row tag interpretation
    tag_columns = [c for c in edges.columns if c not in _NON_TAG_COLS]
    records = edges[tag_columns].to_dict(orient="records")

    modes_strs, directions = [], []
    speed_abs, speed_bas = [], []
    lanes_abs, lanes_bas = [], []
    link_types, names = [], []

    for tags in records:
        tags = {k: v for k, v in tags.items() if v is not None and not _is_nan(v)}
        hw = tags.get("highway")
        if isinstance(hw, list) and hw:
            tags["highway"] = hw[0]
        modes_strs.append(filter_by_modes(compute_modes_string(tags, MODE_RULES), requested_codes))
        directions.append(parse_direction(tags))
        sab, sba = directional_speeds(tags)
        speed_abs.append(sab)
        speed_bas.append(sba)
        lab, lba = directional_lanes(tags)
        lanes_abs.append(lab)
        lanes_bas.append(lba)
        link_types.append(str(tags.get("highway") or "unknown"))
        names.append(tags.get("name"))

    edges = edges.assign(
        modes=modes_strs,
        direction=directions,
        speed_ab=speed_abs,
        speed_ba=speed_bas,
        lanes_ab=lanes_abs,
        lanes_ba=lanes_bas,
        link_type=link_types,
        name=names,
    )

    # ---- Drop links the user did not ask for
    before = len(edges)
    edges = edges[edges["modes"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Mode filter kept {len(edges)} / {before} links")
    if len(edges) == 0:
        raise ImporterError(f"After mode filtering ({modes!r}) no links remain. Try a wider modes set.")

    # ---- Optional clip to model_area polygon
    if clip_to is not None:
        before = len(edges)
        edges = edges[edges.geometry.intersects(clip_to)].reset_index(drop=True)
        logger.info(f"Model-area clip kept {len(edges)} / {before} links")

    # ---- Allocate link_id and _source_id
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    if "osmid" in edges.columns:
        edges["_source_id"] = edges["osmid"].map(lambda v: str(v[0]) if isinstance(v, list) and v else str(v))
    elif "osm_id" in edges.columns:
        edges["_source_id"] = edges["osm_id"].astype(str)
    elif "id" in edges.columns:
        edges["_source_id"] = edges["id"].astype(str)
    else:
        edges["_source_id"] = edges["link_id"].astype(str)

    edges = edges.drop(columns=[c for c in ("u", "v", "key", "osmid") if c in edges.columns])

    # Normalise tag-key column names (colons → underscores)
    rename_map = {
        c: normalise_tag_key(c) for c in edges.columns if c not in _RESERVED_LINK_COLS and normalise_tag_key(c) != c
    }
    if rename_map:
        edges = edges.rename(columns=rename_map)

    # ---- Node staged frame
    used_nodes = set(edges["a_node"]) | set(edges["b_node"])
    node_modes = _compute_node_modes(nodes_gdf["node_id"].to_numpy(), edges)
    nodes_out = gpd.GeoDataFrame(
        {
            "node_id": nodes_gdf["node_id"].astype(np.int64),
            "geometry": nodes_gdf["geometry"],
            "modes": node_modes,
            "_source_id": nodes_gdf["osm_id"].astype(str),
        },
        geometry="geometry", crs="EPSG:4326",
    )
    nodes_out = nodes_out[nodes_out["node_id"].isin(used_nodes)].reset_index(drop=True)
    links_out = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326")
    return StagedNetwork(nodes=nodes_out, links=links_out, source_meta=source_meta)


def _compute_node_modes(node_ids: np.ndarray, edges: pd.DataFrame) -> list:
    """Vectorised: for each node, union of mode chars across every incident link."""
    nodes_col = pd.concat([edges["a_node"], edges["b_node"]], ignore_index=True)
    modes_col = pd.concat([edges["modes"], edges["modes"]], ignore_index=True).map(set)
    per_node = (
        pd.DataFrame({"node": nodes_col, "modes": modes_col})
        .groupby("node")["modes"]
        .agg(lambda s: "".join(sorted(set().union(*s))))
        .to_dict()
    )
    return [per_node.get(int(nid), "") or "c" for nid in node_ids]


def _is_nan(value) -> bool:
    return isinstance(value, float) and value != value  # noqa: PLR0124
