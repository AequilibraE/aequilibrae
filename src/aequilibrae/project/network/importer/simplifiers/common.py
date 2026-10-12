"""Provenance and attribute-map helpers shared by the simplifier backends."""

import json

import geopandas as gpd
from shapely.geometry import LineString

from aequilibrae.project.network.importer.schema.attributes import is_missing, to_jsonable

PROVENANCE_OUT_COL = "source_ids"
SOURCE_ID_COL = "source_id"

PROVENANCE_SCHEMA_VERSION = 1


def build_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    if SOURCE_ID_COL not in links_gdf.columns:
        return {}
    skip = {"a_node", "b_node", "link_id", "geometry", "direction", "distance", PROVENANCE_OUT_COL}
    return {
        str(rec[SOURCE_ID_COL]): {
            str(col): to_jsonable(val)
            for col, val in rec.items()
            if col not in skip and not str(col).startswith("_") and not is_missing(val)
        }
        for rec in links_gdf.to_dict(orient="records")
    }


def build_oriented_source_attr_map(links_gdf: gpd.GeoDataFrame) -> dict:
    out = {}
    for rec in links_gdf.to_dict(orient="records"):
        geom = rec["geometry"]
        source_id = rec.get(SOURCE_ID_COL)
        base_id = str(rec["link_id"] if source_id is None else source_id)
        direction = int(rec["direction"])
        if direction != -1:
            out[f"{base_id}::ab"] = {
                "source_id": base_id,
                "geometry": geom,
                "speed": rec.get("speed_ab"),
                "lanes": rec.get("lanes_ab"),
            }
        if direction != 1:
            out[f"{base_id}::ba"] = {
                "source_id": base_id,
                "geometry": LineString(geom.coords[::-1]) if geom is not None else None,
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
