"""Write staged networks into project Spatialite tables."""

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import JSON_COL, split_attributes
from aequilibrae.project.network.importer.schema.link_types import LinkTypeAllocator
from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.project.network.links_view import LINK_ATTRIBUTES_TABLE, has_link_attributes, links_source
from aequilibrae.utils.db_utils import list_columns
from aequilibrae.utils.sqlite_utils import without_triggers

if TYPE_CHECKING:
    from aequilibrae.project.project import Project

logger = logging.getLogger(__name__)

_OTHER_LINK_TYPE = "other_link_types"


class SpatialiteWriter:
    def __init__(self, project: "Project"):
        self.project = project

    def write(self, net: StagedNetwork) -> None:
        with self.project.db_connection as conn:
            link_cols = list_columns(conn, links_source(conn))
            node_cols = list_columns(conn, "nodes")
            if JSON_COL not in link_cols or JSON_COL not in node_cols:
                raise ImporterError("You must create a new empty project to import a network from OSM/Overture")

            existing = dict(conn.execute("SELECT link_type, link_type_id FROM link_types"))
            links = _fold_excess_link_types(net.links, existing)
            allocator = LinkTypeAllocator(existing)
            new_rows = [
                (allocator.allocate(lt), lt, f"Imported by network importer: {lt}")
                for lt in links["link_type"].dropna().astype(str).unique()
                if lt not in existing
            ]
            conn.executemany("INSERT INTO link_types (link_type_id, link_type, description) VALUES (?, ?, ?)", new_rows)

            # Without triggers, so link attributes are written to their own table explicitly
            base_link_cols = set(list_columns(conn, "links")) if has_link_attributes(conn) else None
            with without_triggers(conn):
                _insert(conn, "nodes", net.nodes, node_cols)
                _insert(conn, "links", links, link_cols, base_link_cols)


def _fold_excess_link_types(links: gpd.GeoDataFrame, existing: dict) -> gpd.GeoDataFrame:
    """Fold rare link types when single-character IDs would be exhausted."""
    free_slots = LinkTypeAllocator.count_free_slots(existing)
    counts = links["link_type"].dropna().astype(str).value_counts()
    new_types = counts[~counts.index.isin(existing)]
    if len(new_types) <= free_slots:
        return links

    keep_n = max(free_slots - (_OTHER_LINK_TYPE not in existing), 0)
    keep = new_types.drop(_OTHER_LINK_TYPE, errors="ignore").index[:keep_n]
    fold = new_types.index.difference([*keep, _OTHER_LINK_TYPE])
    logger.warning(
        "Number of new link types (%d) exceeds the available single-character ids (%d). "
        "Folding the %d least-frequent types into '%s': %s",
        len(new_types),
        free_slots,
        len(fold),
        _OTHER_LINK_TYPE,
        ", ".join(sorted(fold)),
    )
    return links.assign(link_type=links["link_type"].where(~links["link_type"].isin(fold), _OTHER_LINK_TYPE))


def _insert(conn, table: str, gdf: gpd.GeoDataFrame, table_cols: list, base_cols: set | None = None) -> None:
    """Insert a staged layer. When ``base_cols`` is given, the columns outside it go to link_attributes."""
    direct, extra_json = split_attributes(gdf, table_cols)
    direct[JSON_COL] = extra_json
    col_names = [c for c in direct.columns if c != "geometry"]
    if base_cols is None:
        _insert_rows(conn, table, direct, col_names, with_geometry=True)
        return

    _insert_rows(conn, table, direct, [c for c in col_names if c in base_cols], with_geometry=True)
    attribute_names = ["link_id"] + [c for c in col_names if c not in base_cols]
    _insert_rows(conn, LINK_ATTRIBUTES_TABLE, direct, attribute_names, with_geometry=False)


def _insert_rows(conn, table: str, frame: gpd.GeoDataFrame, col_names: list, with_geometry: bool) -> None:
    columns = ", ".join(col_names + ["geometry"] if with_geometry else col_names)
    placeholders = ",".join(["?"] * len(col_names) + (["GeomFromWKB(?, 4326)"] if with_geometry else []))
    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    values = frame[col_names].astype(object).where(pd.notna(frame[col_names]), None)
    records = values.itertuples(index=False, name=None)
    if not with_geometry:
        conn.executemany(sql, records)
        return
    wkbs = frame.geometry.to_wkb(output_dimension=2)
    conn.executemany(sql, (r + (wkb,) for r, wkb in zip(records, wkbs, strict=True)))
