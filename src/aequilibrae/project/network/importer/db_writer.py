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

_OTHER_LINK_TYPE = "other_link_types"


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

            net = self._fold_excess_link_types(conn, net)
            self._ensure_link_types(conn, net.links["link_type"].dropna().astype(str).unique())

            triggers_before = _count_triggers(conn)
            remove_triggers(conn, "network")
            try:
                self._insert_nodes(conn, net.nodes, node_cols)
                self._insert_links(conn, net.links, link_cols)
            finally:
                add_triggers(conn, "network")
                self._verify_triggers_restored(conn, triggers_before)

    @staticmethod
    def _verify_triggers_restored(conn, expected_min: int) -> None:
        """Fail loudly if bulk-insert trigger stripping left the schema weakened."""
        restored = _count_triggers(conn)
        if restored < expected_min:
            raise ImporterError(
                "Network triggers were not fully restored after the bulk insert "
                f"(found {restored}, expected at least {expected_min}). The project schema may be "
                "in an inconsistent state; recreate the project and re-run the import."
            )

    def _fold_excess_link_types(self, conn, net: StagedNetwork) -> StagedNetwork:
        """Bucket the least-frequent link types into ``other_link_types`` when the
        number of distinct types would exceed the single-character ``link_type_id``
        alphabet (the schema enforces ``LENGTH(link_type_id) == 1``).

        We keep the most-used link types as first-class entries and collapse the
        long tail of rare types into a single catch-all so a rich import never
        crashes with an alphabet-exhaustion ``RuntimeError`` mid-write.
        """
        existing = {row[0]: row[1] for row in conn.execute("SELECT link_type, link_type_id FROM link_types").fetchall()}
        free_slots = LinkTypeAllocator.count_free_slots(existing)

        link_types = net.links["link_type"].dropna().astype(str)
        new_types = [lt for lt in link_types.unique() if lt not in existing]
        if len(new_types) <= free_slots:
            return net

        # Reserve one slot for the catch-all bucket itself.
        keep_n = max(free_slots - 1, 0)
        counts = link_types[link_types.isin(new_types)].value_counts()
        keep = set(counts.index[:keep_n])
        fold = [lt for lt in new_types if lt not in keep]

        if not fold:
            return net

        logger.warning(
            "Number of new link types (%d) exceeds the available single-character ids (%d). "
            "Folding the %d least-frequent types into '%s': %s",
            len(new_types),
            free_slots,
            len(fold),
            _OTHER_LINK_TYPE,
            ", ".join(sorted(fold)),
        )

        links = net.links.copy()
        fold_set = set(fold)
        links["link_type"] = links["link_type"].where(~links["link_type"].isin(fold_set), _OTHER_LINK_TYPE)
        return StagedNetwork(
            nodes=net.nodes,
            links=links,
            crs_geo=net.crs_geo,
            source_meta=net.source_meta,
        )

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


def _count_triggers(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0])


def _to_records(direct: gpd.GeoDataFrame, col_names: list) -> list:
    """Per-row tuples (NaN → None) for the given columns, via vectorised conversion."""
    if not col_names:
        return [() for _ in range(len(direct))]
    sub = direct[col_names].astype(object).where(pd.notna(direct[col_names]), None)
    return list(sub.itertuples(index=False, name=None))
