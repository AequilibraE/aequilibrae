"""Write staged networks into project Spatialite tables."""

import logging
from typing import TYPE_CHECKING, Iterable

import geopandas as gpd
import pandas as pd

from aequilibrae.project.project_creation import add_triggers, remove_triggers
from aequilibrae.utils.db_utils import commit_and_close, list_columns

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.attributes import JSON_COL, split_attributes
from aequilibrae.project.network.importer.schema.link_types import LinkTypeAllocator
from aequilibrae.project.network.importer.staged_network import StagedNetwork

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


class SpatialiteWriter:

    def __init__(self, project: "Project"):
        self.project = project
        self.path = project.path_to_file

    def write(self, net: StagedNetwork) -> None:
        with commit_and_close(self.path, spatial=True) as conn:
            link_cols = list_columns(conn, "links")
            node_cols = list_columns(conn, "nodes")
            if JSON_COL not in link_cols or JSON_COL not in node_cols:
                raise ImporterError("You must create a new empty project to import a network from OSM/Overture")

            self._ensure_link_types(conn, net.links["link_type"].dropna().astype(str).unique())

            remove_triggers(conn, "network")
            try:
                self._insert_nodes(conn, net.nodes, node_cols)
                self._insert_links(conn, net.links, link_cols)
            finally:
                add_triggers(conn, "network")

    def _ensure_link_types(self, conn, link_types: Iterable[str]) -> None:
        existing = {row[0]: row[1] for row in conn.execute("SELECT link_type, link_type_id FROM link_types").fetchall()}
        allocator = LinkTypeAllocator(existing=existing)
        new_rows = []
        for lt in link_types:
            if lt in existing:
                continue
            code = allocator.allocate(lt)
            new_rows.append((code, lt, f"Imported by network importer: {lt}"))
        if new_rows:
            conn.executemany(
                "INSERT INTO link_types (link_type_id, link_type, description) VALUES (?, ?, ?)",
                new_rows,
            )

    def _insert_nodes(self, conn, nodes_gdf: gpd.GeoDataFrame, table_cols: list) -> None:
        direct, extra_json = split_attributes(nodes_gdf, table_cols)
        direct = direct.assign(**{JSON_COL: extra_json})

        if "geometry" not in direct.columns:
            raise ImporterError("nodes IR missing geometry column")

        if "is_centroid" in table_cols and "is_centroid" not in direct.columns:
            direct["is_centroid"] = 0

        col_names = [c for c in direct.columns if c != "geometry"]
        placeholders = ",".join(["?"] * len(col_names))
        sql = f"INSERT INTO nodes ({', '.join(col_names)}, geometry) VALUES ({placeholders}, MakePoint(?, ?, 4326))"
        xs = direct.geometry.x.to_numpy()
        ys = direct.geometry.y.to_numpy()
        records = _to_records(direct, col_names)
        conn.executemany(sql, [r + (float(x), float(y)) for r, x, y in zip(records, xs, ys, strict=True)])

    def _insert_links(self, conn, links_gdf: gpd.GeoDataFrame, table_cols: list) -> None:
        direct, extra_json = split_attributes(links_gdf, table_cols)
        direct = direct.assign(**{JSON_COL: extra_json})

        col_names = [c for c in direct.columns if c != "geometry"]
        placeholders = ",".join(["?"] * len(col_names))
        sql = f"INSERT INTO links ({', '.join(col_names)}, geometry) VALUES ({placeholders}, GeomFromWKB(?, 4326))"
        wkbs = direct.geometry.to_wkb()
        records = _to_records(direct, col_names)
        conn.executemany(sql, [r + (wkb,) for r, wkb in zip(records, wkbs, strict=True)])


def _to_records(direct: gpd.GeoDataFrame, col_names: list) -> list:
    """Per-row tuples (NaN → None) for the given columns, via vectorised conversion."""
    if not col_names:
        return [() for _ in range(len(direct))]
    sub = direct[col_names].astype(object).where(pd.notna(direct[col_names]), None)
    return list(sub.itertuples(index=False, name=None))
