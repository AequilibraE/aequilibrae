"""Overture segment + connector → staged network.

Pipeline:
  1. Build nodes from the connectors GeoDataFrame.
  2. For each segment, sort connectors by ``at`` and split via shapely
     ``substring`` at every intermediate connector.
  3. Derive mode / direction / speed / link_type / distance from the
     segment's fields; pass everything else through as free-form columns.
     The full rule arrays are always preserved.
"""

import json
import logging
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import substring

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import to_jsonable
from aequilibrae.project.network.importer.schema.modes import filter_by_modes
from aequilibrae.project.network.importer.sources.osm.tags_to_ir import MODE_CODE
from aequilibrae.project.network.importer.staged_network import StagedNetwork

logger = logging.getLogger(__name__)


_NODE_START = 10000
_NON_ROAD_SUBTYPES = {"rail", "water"}

_MOTORISED_CLASSES = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "living_street", "unclassified", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "road",
})
_MIXED_TRAFFIC_CLASSES = frozenset({
    "residential", "living_street", "unclassified", "tertiary", "secondary", "primary",
})
_HIGHWAY_LIKE_CLASSES = frozenset({"trunk", "motorway", "trunk_link", "motorway_link"})
_PEDESTRIAN_CLASSES = frozenset({"footway", "pedestrian", "path", "sidewalk", "steps", "crosswalk"})
_BICYCLE_FRIENDLY_PED_CLASSES = frozenset({"path", "crosswalk"})
_BICYCLE_CLASSES = frozenset({"cycleway", "bicycle_path"})


def build_staged_from_overture(
    *,
    connectors: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    modes: Sequence[str],
    source_meta: dict,
) -> StagedNetwork:
    if len(segments) == 0:
        raise ImporterError("Overture returned no segments in the requested area")
    if len(connectors) == 0:
        raise ImporterError("Overture returned no connectors in the requested area")

    requested_codes = {MODE_CODE[m] for m in modes if m in MODE_CODE}
    if not requested_codes:
        raise ImporterError(f"None of the requested modes {modes!r} match the configured modes {sorted(MODE_CODE)}")

    # ---- Connector → AeQ node_id map
    connectors = connectors.to_crs("EPSG:4326").dropna(subset=["geometry"]).reset_index(drop=True)
    connectors["node_id"] = np.arange(_NODE_START, _NODE_START + len(connectors), dtype=np.int64)
    connectors["_source_id"] = connectors["id"].astype(str)
    gers_to_node = dict(zip(connectors["_source_id"], connectors["node_id"]))

    # ---- Segments → sub-link rows
    segments = segments.to_crs("EPSG:4326")

    # to_dict(records) preserves the actual column names (incl. reserved words
    # like 'class'); itertuples mangles them into _N.
    link_rows = []
    seg_records = segments.drop(columns=["geometry"]).to_dict(orient="records")
    for seg, geom in zip(seg_records, segments.geometry):
        link_rows.extend(_segment_to_links(seg, geom, gers_to_node, requested_codes))

    if not link_rows:
        raise ImporterError(f"After mode filtering ({modes!r}) no Overture links remain")

    links_gdf = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")
    utm = links_gdf.geometry.estimate_utm_crs()
    links_gdf["distance"] = links_gdf.geometry.to_crs(utm).length.astype(float)
    links_gdf = links_gdf[links_gdf["distance"] > 0].reset_index(drop=True)
    links_gdf["link_id"] = np.arange(1, len(links_gdf) + 1, dtype=np.int64)

    used = set(links_gdf["a_node"]) | set(links_gdf["b_node"])
    nodes_gdf = connectors[connectors["node_id"].isin(used)].reset_index(drop=True)
    nodes_out = gpd.GeoDataFrame(
        {
            "node_id": nodes_gdf["node_id"].astype(np.int64),
            "geometry": nodes_gdf["geometry"],
            "modes": _compute_node_modes(nodes_gdf["node_id"].to_numpy(), links_gdf),
            "_source_id": nodes_gdf["_source_id"],
            "gers_id": nodes_gdf["_source_id"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    return StagedNetwork(nodes=nodes_out, links=links_gdf, source_meta=source_meta)


def _segment_to_links(seg: dict, geom, gers_to_node: dict, requested_codes: set) -> list:
    if geom is None or geom.is_empty:
        return []
    pairs = _parse_connectors_field(seg.get("connectors"))
    if len(pairs) < 2 or any(cid not in gers_to_node for cid, _ in pairs):
        return []

    filtered_modes = filter_by_modes(_modes_for_segment(seg), requested_codes)
    if not filtered_modes:
        return []

    direction = _direction_for_segment(seg)
    speed_ab, speed_ba = _speeds_for_segment(seg, direction)
    link_type = str(seg.get("class") or "unknown")
    sid = str(seg.get("id") or "")
    free_attrs = _free_attrs(seg)

    out = []
    for (cid_a, at_a), (cid_b, at_b) in zip(pairs[:-1], pairs[1:], strict=True):
        if at_b <= at_a:
            continue
        sub = substring(geom, at_a, at_b, normalized=True)
        if sub.is_empty:
            continue
        out.append({
            "a_node": gers_to_node[cid_a],
            "b_node": gers_to_node[cid_b],
            "direction": direction,
            "modes": filtered_modes,
            "link_type": link_type,
            "name": seg.get("primary_name") or seg.get("names.primary"),
            "speed_ab": speed_ab,
            "speed_ba": speed_ba,
            "lanes_ab": None,
            "lanes_ba": None,
            "geometry": sub,
            "_source_id": sid,
            "gers_id": sid,
            **free_attrs,
        })
    return out


def _parse_connectors_field(value) -> list:
    """Overture's ``connectors`` array → list of ``(connector_id, at)`` sorted by ``at``."""
    if value is None:
        return []
    pairs = []
    for item in value:
        if item is None:
            continue
        cid = item.get("connector_id") or item.get("id")
        if cid is None:
            continue
        at = item.get("at")
        pairs.append((str(cid), float(at) if at is not None else 0.0))
    pairs.sort(key=lambda p: p[1])
    return pairs


def _modes_for_segment(seg: dict) -> str:
    """Compute the modes string for an Overture segment using access semantics."""
    subtype = str(seg.get("subtype") or "").lower()
    if subtype in _NON_ROAD_SUBTYPES:
        return ""
    cls = str(seg.get("class") or "").lower()

    out: set = set()
    if subtype in ("road", "") and cls in _MOTORISED_CLASSES:
        out.add(MODE_CODE["car"])
        if cls not in ("motorway", "motorway_link"):
            out.add(MODE_CODE["transit"])
        if cls in _MIXED_TRAFFIC_CLASSES:
            out.add(MODE_CODE["bicycle"])
            out.add(MODE_CODE["walk"])
        elif cls not in _HIGHWAY_LIKE_CLASSES:
            out.add(MODE_CODE["walk"])

    if cls in _PEDESTRIAN_CLASSES:
        out.add(MODE_CODE["walk"])
        if cls in _BICYCLE_FRIENDLY_PED_CLASSES:
            out.add(MODE_CODE["bicycle"])
    if cls in _BICYCLE_CLASSES:
        out.add(MODE_CODE["bicycle"])
    return "".join(sorted(out))


def _direction_for_segment(seg: dict) -> int:
    """Derive AeQ direction from ``access_restrictions`` (global entries only)."""
    restrictions = seg.get("access_restrictions")
    if restrictions is None:
        return 0
    has_forward_deny = False
    has_backward_deny = False
    for rule in restrictions:
        if rule is None or str(rule.get("access_type") or "").lower() != "denied":
            continue
        when = rule.get("when") or {}
        heading = (when.get("heading") if when else None) or rule.get("heading")
        if heading == "forward":
            has_forward_deny = True
        elif heading == "backward":
            has_backward_deny = True
    if has_forward_deny and not has_backward_deny:
        return -1
    if has_backward_deny and not has_forward_deny:
        return 1
    return 0


def _speeds_for_segment(seg: dict, direction: int) -> tuple:
    """Parse ``speed_limits[]`` global entry (no scoping) into (ab, ba) km/h."""
    limits = seg.get("speed_limits")
    if limits is None:
        return (None, None)
    speed = None
    for rule in limits:
        if rule is None:
            continue
        if rule.get("between") is not None or rule.get("when") is not None:
            continue
        ms = rule.get("max_speed")
        if ms is None:
            continue
        value = ms.get("value")
        if value is None:
            continue
        value = float(value)
        if "mph" in str(ms.get("unit") or "km/h").lower():
            value *= 1.609344
        speed = value
        break
    if speed is None:
        return (None, None)
    if direction == 1:
        return (speed, None)
    if direction == -1:
        return (None, speed)
    return (speed, speed)


_PASS_THROUGH_KEYS = (
    "subtype", "class", "subclass", "road_flags", "road_surface",
    "level_rules", "routes", "destinations", "width_rules", "names", "primary_name",
)
_RULE_ARRAY_KEYS = ("access_restrictions", "prohibited_transitions", "subclass_rules", "speed_limits")


def _free_attrs(seg: dict) -> dict:
    """Pass-through Overture properties as free-form staged-network columns."""
    out = {}
    for key in _PASS_THROUGH_KEYS:
        value = seg.get(key)
        if value is None:
            continue
        value = value.tolist() if isinstance(value, np.ndarray) else value
        out[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(to_jsonable(value), default=str)
    for key in _RULE_ARRAY_KEYS:
        value = seg.get(key)
        if value is None:
            continue
        value = value.tolist() if isinstance(value, np.ndarray) else value
        out[key] = json.dumps(to_jsonable(value), default=str)
    return out


def _compute_node_modes(node_ids: np.ndarray, links: gpd.GeoDataFrame) -> list:
    """Vectorised union of mode chars per node."""
    nodes_col = pd.concat([links["a_node"], links["b_node"]], ignore_index=True)
    modes_col = pd.concat([links["modes"], links["modes"]], ignore_index=True).map(set)
    per_node = (
        pd.DataFrame({"node": nodes_col, "modes": modes_col})
        .groupby("node")["modes"]
        .agg(lambda s: "".join(sorted(set().union(*s))))
        .to_dict()
    )
    return [per_node.get(int(nid), "") or "c" for nid in node_ids]
