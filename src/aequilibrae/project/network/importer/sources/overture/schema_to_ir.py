"""Overture segment + connector → IR.

Plan §6.2. The pipeline is:
  1. Build nodes from the connectors GeoDataFrame.
  2. For each segment, sort connectors by ``at`` and split via shapely
     ``substring`` at every intermediate connector.
  3. Derive mode / direction / speed / link_type / distance from the segment's
     fields; pass everything else through as free-form columns. The full rule
     arrays (``access_restrictions``, ``prohibited_transitions``,
     ``subclass_rules``, ``speed_limits``) are always preserved per plan §1.3
     rule 7.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from ...exceptions import ImporterError
from ...ir import RoutableNetwork
from ...schema.modes import filter_by_modes
from ..osm.tags_to_ir import MODE_CODE

logger = logging.getLogger(__name__)


_NODE_START = 10000

# Subtypes that aren't routable as motor vehicles even when 'car' is requested
_NON_ROAD_SUBTYPES = {"rail", "water"}


def build_ir_from_overture(
    *,
    connectors: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    modes: Sequence[str],
    source_meta: dict,
) -> RoutableNetwork:
    if len(segments) == 0:
        raise ImporterError("Overture returned no segments in the requested area")
    if len(connectors) == 0:
        raise ImporterError("Overture returned no connectors in the requested area")

    requested_codes = {MODE_CODE[m] for m in modes if m in MODE_CODE}
    if not requested_codes:
        raise ImporterError(
            f"None of the requested modes {modes!r} match the configured modes "
            f"{sorted(MODE_CODE)}"
        )

    # ---- Build the connector → AeQ node_id map
    connectors = connectors.copy()
    if str(connectors.crs).upper() != "EPSG:4326":
        connectors = connectors.to_crs("EPSG:4326")
    connectors = connectors.dropna(subset=["geometry"]).reset_index(drop=True)
    connectors["node_id"] = np.arange(
        _NODE_START, _NODE_START + len(connectors), dtype=np.int64
    )
    connectors["_source_id"] = connectors["id"].astype(str)
    gers_to_node = dict(zip(connectors["id"].astype(str), connectors["node_id"]))

    # ---- Walk segments → sub-link rows
    segments = segments.copy()
    if str(segments.crs).upper() != "EPSG:4326":
        segments = segments.to_crs("EPSG:4326")

    link_rows: list[dict] = []
    for _, seg in segments.iterrows():
        link_rows.extend(_segment_to_links(seg, gers_to_node, requested_codes))

    if not link_rows:
        raise ImporterError(
            f"After mode filtering ({modes!r}) no Overture links remain"
        )

    links_gdf = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")
    links_gdf["link_id"] = np.arange(1, len(links_gdf) + 1, dtype=np.int64)

    # ---- distance via auto-UTM
    utm = links_gdf.geometry.estimate_utm_crs()
    links_gdf["distance"] = links_gdf.geometry.to_crs(utm).length.astype(float)
    # zero-length sub-segments are noise; drop them
    links_gdf = links_gdf[links_gdf["distance"] > 0].reset_index(drop=True)
    links_gdf["link_id"] = np.arange(1, len(links_gdf) + 1, dtype=np.int64)

    # ---- Nodes: keep only ones referenced by surviving links
    used = set(links_gdf["a_node"]).union(links_gdf["b_node"])
    nodes_gdf = connectors[connectors["node_id"].isin(used)].copy().reset_index(drop=True)
    nodes_out = gpd.GeoDataFrame(
        {
            "node_id": nodes_gdf["node_id"].astype(np.int64),
            "geometry": nodes_gdf["geometry"],
            "modes": _compute_node_modes(nodes_gdf["node_id"], links_gdf),
            "_source_id": nodes_gdf["_source_id"],
            # Pass-through: GERS id will land in other_attributes alongside source_id
            "gers_id": nodes_gdf["_source_id"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    return RoutableNetwork(nodes=nodes_out, links=links_gdf, source_meta=source_meta)


def _segment_to_links(seg, gers_to_node, requested_codes):
    geom = seg.geometry
    if geom is None or geom.is_empty:
        return []
    connectors_field = seg.get("connectors")
    pairs = _parse_connectors_field(connectors_field)
    if len(pairs) < 2:
        return []

    # Map every connector id to an AeQ node id; if missing (e.g. clipped out),
    # skip the segment.
    for cid, _at in pairs:
        if cid not in gers_to_node:
            return []

    out = []
    raw_modes = _modes_for_segment(seg)
    filtered_modes = filter_by_modes(raw_modes, requested_codes)
    if not filtered_modes:
        return []

    direction = _direction_for_segment(seg)
    speed_ab, speed_ba = _speeds_for_segment(seg, direction)
    link_type = str(seg.get("class") or "unknown")
    free_attrs = _free_attrs(seg)

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
            "name": seg.get("names.primary") or seg.get("primary_name"),
            "speed_ab": speed_ab,
            "speed_ba": speed_ba,
            "lanes_ab": None,
            "lanes_ba": None,
            "geometry": sub,
            "_source_id": str(seg.get("id") or ""),
            "gers_id": str(seg.get("id") or ""),
            **free_attrs,
        })
    return out


def _parse_connectors_field(value) -> list[tuple[str, float]]:
    """Overture's ``connectors`` array → list of ``(connector_id, at)`` sorted by ``at``."""
    if value is None:
        return []
    # value may be a numpy array, list of dicts, or list of pyarrow structs
    try:
        items = list(value)
    except TypeError:
        return []
    pairs = []
    for item in items:
        if item is None:
            continue
        cid = None
        at = None
        if hasattr(item, "get"):
            cid = item.get("connector_id") or item.get("id")
            at = item.get("at")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            cid, at = str(item[0]), float(item[1])
        if cid is None:
            continue
        try:
            at = float(at) if at is not None else 0.0
        except (TypeError, ValueError):
            continue
        pairs.append((str(cid), at))
    pairs.sort(key=lambda p: p[1])
    return pairs


def _modes_for_segment(seg) -> str:
    """Compute the modes string for an Overture segment using access semantics."""
    subtype = str(seg.get("subtype") or "").lower()
    if subtype in _NON_ROAD_SUBTYPES:
        return ""
    cls = str(seg.get("class") or "").lower()

    out: set[str] = set()
    # Car / transit on motorised roads
    if subtype in ("road", ""):
        if cls in ("motorway", "trunk", "primary", "secondary", "tertiary",
                   "residential", "living_street", "unclassified", "service",
                   "motorway_link", "trunk_link", "primary_link", "secondary_link",
                   "tertiary_link", "road"):
            out.add(MODE_CODE["car"])
            if cls not in ("motorway", "motorway_link"):
                out.add(MODE_CODE["transit"])
            if cls in ("residential", "living_street", "unclassified", "tertiary",
                       "secondary", "primary"):
                out.add(MODE_CODE["bicycle"])
                out.add(MODE_CODE["walk"])
            elif cls in ("trunk", "motorway", "trunk_link", "motorway_link"):
                pass
            else:
                out.add(MODE_CODE["walk"])
    # Pedestrian + cycle infrastructure
    if cls in ("footway", "pedestrian", "path", "sidewalk", "steps", "crosswalk"):
        out.add(MODE_CODE["walk"])
        if cls in ("path", "crosswalk"):
            out.add(MODE_CODE["bicycle"])
    if cls in ("cycleway", "bicycle_path"):
        out.add(MODE_CODE["bicycle"])
    return "".join(sorted(out))


def _direction_for_segment(seg) -> int:
    """Derive AeQ direction from ``access_restrictions`` (global entries only)."""
    restrictions = seg.get("access_restrictions")
    if restrictions is None:
        return 0
    try:
        rules = list(restrictions)
    except TypeError:
        return 0
    has_forward_deny = False
    has_backward_deny = False
    for rule in rules:
        if rule is None or not hasattr(rule, "get"):
            continue
        access_type = str(rule.get("access_type") or "").lower()
        when = rule.get("when") or {}
        heading = (when.get("heading") if hasattr(when, "get") else None) or rule.get("heading")
        if access_type != "denied":
            continue
        if heading == "forward":
            has_forward_deny = True
        elif heading == "backward":
            has_backward_deny = True
    if has_forward_deny and not has_backward_deny:
        return -1
    if has_backward_deny and not has_forward_deny:
        return 1
    return 0


def _speeds_for_segment(seg, direction: int) -> tuple[float | None, float | None]:
    """Parse ``speed_limits[]`` global entry (no scoping) into (ab, ba) km/h."""
    limits = seg.get("speed_limits")
    if limits is None:
        return (None, None)
    try:
        rules = list(limits)
    except TypeError:
        return (None, None)
    speed = None
    for rule in rules:
        if rule is None or not hasattr(rule, "get"):
            continue
        if rule.get("between") is not None or rule.get("when") is not None:
            continue
        ms = rule.get("max_speed")
        if ms is None:
            continue
        if hasattr(ms, "get"):
            value = ms.get("value")
            unit = str(ms.get("unit") or "km/h").lower()
        else:
            value, unit = ms, "km/h"
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if "mph" in unit:
            value = value * 1.609344
        speed = value
        break
    if speed is None:
        return (None, None)
    if direction == 1:
        return (speed, None)
    if direction == -1:
        return (None, speed)
    return (speed, speed)


def _free_attrs(seg) -> dict:
    """Pass-through Overture properties as free-form IR columns.

    The committer will route them to real columns where they match or to
    ``other_attributes`` otherwise. Rule arrays are JSON-stringified so they
    fit the ``other_attributes`` column type cleanly.
    """
    import json

    out = {}
    pass_keys = (
        "subtype", "class", "subclass", "road_flags", "road_surface",
        "level_rules", "routes", "destinations", "width_rules",
        "names", "primary_name",
    )
    for key in pass_keys:
        if key in seg and seg[key] is not None:
            value = seg[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            out[key] = value if isinstance(value, (str, int, float, bool)) \
                else json.dumps(_to_jsonable(value), default=str)
    # The rule arrays are ALWAYS preserved per plan §1.3 rule 7
    for key in ("access_restrictions", "prohibited_transitions",
                "subclass_rules", "speed_limits"):
        if key in seg and seg[key] is not None:
            value = seg[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            out[key] = json.dumps(_to_jsonable(value), default=str)
    return out


def _to_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    return str(value)


def _compute_node_modes(node_ids: Iterable[int], links: gpd.GeoDataFrame) -> list[str]:
    by_node: dict[int, set[str]] = {int(nid): set() for nid in node_ids}
    for _, row in links.iterrows():
        a = int(row["a_node"])
        b = int(row["b_node"])
        for ch in str(row["modes"]):
            if ch:
                by_node.setdefault(a, set()).add(ch)
                by_node.setdefault(b, set()).add(ch)
    return ["".join(sorted(by_node.get(int(nid), set()))) or "c" for nid in node_ids]
