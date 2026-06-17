"""Schema-aware committer that writes a ``RoutableNetwork`` into spatialite.

Per plan §4.4 the committer is **strictly non-schema-modifying**: it issues
**no ``ALTER TABLE`` statements at all**. Source-specific tags / properties /
free-form attributes are JSON-encoded into the existing ``other_attributes``
column on ``links`` / ``nodes``; if that column is missing the import fails
with a documented actionable error.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from aequilibrae.project.project_creation import add_triggers, remove_triggers
from aequilibrae.utils.db_utils import commit_and_close, list_columns

from .exceptions import ImporterError
from .ir import RoutableNetwork
from .schema.attributes import JSON_COL, split_attributes
from .schema.link_types import LinkTypeAllocator

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


class SpatialiteWriter:
    """Atomic committer for an IR.

    Issues zero ``ALTER TABLE``s. Validates that ``other_attributes`` exists
    on both ``links`` and ``nodes``; raises if missing.
    """

    def __init__(self, project: "Project"):
        self.project = project
        self.path = project.path_to_file

    def write(self, net: RoutableNetwork) -> None:
        with commit_and_close(self.path, spatial=True) as conn:
            link_cols = list_columns(conn, "links")
            node_cols = list_columns(conn, "nodes")
            if JSON_COL not in link_cols:
                raise ImporterError(
                    "links table is missing the 'other_attributes' column. "
                    "Recreate the project with the current AequilibraE version, or add "
                    "the column manually: ALTER TABLE links ADD COLUMN other_attributes TEXT;"
                )
            if JSON_COL not in node_cols:
                raise ImporterError(
                    "nodes table is missing the 'other_attributes' column. "
                    "Recreate the project with the current AequilibraE version, or add "
                    "the column manually: ALTER TABLE nodes ADD COLUMN other_attributes TEXT;"
                )

            # Ensure link_types referenced by links exist in the link_types table.
            # This is the only schema-management we do (insert into link_types,
            # never ALTER any table).
            self._ensure_link_types(conn, net.links["link_type"].dropna().astype(str).unique())

            remove_triggers(conn, "network")
            try:
                self._insert_nodes(conn, net.nodes, node_cols)
                self._insert_links(conn, net.links, link_cols)
            finally:
                add_triggers(conn, "network")

    # ---------- link_types ----------

    def _ensure_link_types(self, conn, link_types: Iterable[str]) -> None:
        existing = {
            row[0]: row[1]
            for row in conn.execute("SELECT link_type, link_type_id FROM link_types").fetchall()
        }
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

    # ---------- nodes ----------

    def _insert_nodes(self, conn, nodes_gdf: gpd.GeoDataFrame, table_cols: list[str]) -> None:
        direct, extra_json = split_attributes(nodes_gdf, table_cols)
        direct = direct.copy()
        direct[JSON_COL] = extra_json

        # Geometry: insert via MakePoint
        if "geometry" not in direct.columns:
            raise ImporterError("nodes IR missing geometry column")

        col_names = [c for c in direct.columns if c != "geometry"]
        # If is_centroid not provided, default to 0
        if "is_centroid" in table_cols and "is_centroid" not in col_names:
            direct["is_centroid"] = 0
            col_names.append("is_centroid")

        placeholders = ",".join(["?"] * len(col_names))
        sql = (
            f"INSERT INTO nodes ({', '.join(col_names)}, geometry) "
            f"VALUES ({placeholders}, MakePoint(?, ?, 4326))"
        )

        rows = []
        for _, row in direct.iterrows():
            geom: BaseGeometry = row["geometry"]
            values = []
            for c in col_names:
                values.append(_normalise_value(row[c]))
            values.append(float(geom.x))
            values.append(float(geom.y))
            rows.append(values)
        conn.executemany(sql, rows)

    # ---------- links ----------

    def _insert_links(self, conn, links_gdf: gpd.GeoDataFrame, table_cols: list[str]) -> None:
        direct, extra_json = split_attributes(links_gdf, table_cols)
        direct = direct.copy()
        direct[JSON_COL] = extra_json

        col_names = [c for c in direct.columns if c != "geometry"]
        placeholders = ",".join(["?"] * len(col_names))
        sql = (
            f"INSERT INTO links ({', '.join(col_names)}, geometry) "
            f"VALUES ({placeholders}, GeomFromWKB(?, 4326))"
        )

        rows = []
        for _, row in direct.iterrows():
            geom: BaseGeometry = row["geometry"]
            values = []
            for c in col_names:
                values.append(_normalise_value(row[c]))
            values.append(geom.wkb)
            rows.append(values)
        conn.executemany(sql, rows)


def _normalise_value(value):
    """Coerce pandas / numpy scalars to plain Python types for sqlite3."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float, str, bytes, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    # numpy scalar?
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)
