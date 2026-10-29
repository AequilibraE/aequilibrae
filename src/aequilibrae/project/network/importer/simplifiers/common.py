"""Provenance and attribute-map helpers shared by the simplifier backends."""

import json

import geopandas as gpd
from shapely.geometry import LineString

from aequilibrae.project.network.importer.schema.attributes import is_missing, to_jsonable

PROVENANCE_OUT_COL = "source_ids"
SOURCE_ID_COL = "source_id"

# Bump when the structure of the provenance payload changes so downstream
# deserialisers can detect/adapt to the format.
PROVENANCE_SCHEMA_VERSION = 1


def build_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    if SOURCE_ID_COL not in links_gdf.columns:
        return {}
    skip = {
        "a_node",
        "b_node",
        "link_id",
        "geometry",
        "direction",
        "distance",
        PROVENANCE_OUT_COL,
    }
    out = {}
    for rec in links_gdf.to_dict(orient="records"):
        attrs = {}
        for col, val in rec.items():
            if is_missing(val) or col in skip or str(col).startswith("_"):
                continue
            attrs[str(col)] = to_jsonable(val)
        out[str(rec[SOURCE_ID_COL])] = attrs
    return out


def build_oriented_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    out = {}
    for rec, geom in zip(links_gdf.to_dict(orient="records"), links_gdf.geometry, strict=True):
        base_id = str(rec.get(SOURCE_ID_COL) if rec.get(SOURCE_ID_COL) is not None else rec["link_id"])
        direction = int(rec["direction"])
        if direction in (0, 1):
            out[f"{base_id}::ab"] = {
                "source_id": base_id,
                "geometry": geom,
                "speed": rec.get("speed_ab"),
                "lanes": rec.get("lanes_ab"),
            }
        if direction in (0, -1):
            rev_geom = LineString(geom.coords[::-1]) if geom is not None else None
            out[f"{base_id}::ba"] = {
                "source_id": base_id,
                "geometry": rev_geom,
                "speed": rec.get("speed_ba"),
                "lanes": rec.get("lanes_ba"),
            }
    return out


def build_provenance(source_ids: list, src_attrs: dict):
    if not source_ids:
        return None
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "sources": {sid: src_attrs.get(sid, {}) for sid in source_ids},
    }
    return json.dumps(payload, separators=(",", ":"), default=str)
