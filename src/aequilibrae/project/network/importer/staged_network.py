"""Staged network value passed between sources, simplifiers, and the writer.

A ``StagedNetwork`` is the in-flight (nodes, links) pair plus the source
provenance dict that the orchestrator carries from a ``Source`` through the
optional simplifier and into the spatialite committer.

Sources are encouraged to surface every attribute they extract as free-form
columns; the committer decides per column whether it lands in a real DB
column or is JSON-encoded into ``other_attributes``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np

from .exceptions import StagedNetworkValidationError

if TYPE_CHECKING:
    import networkx as nx


_REQUIRED_NODE_COLS = ("node_id", "geometry", "modes")
_REQUIRED_LINK_COLS = (
    "link_id",
    "a_node",
    "b_node",
    "direction",
    "modes",
    "link_type",
    "distance",
    "geometry",
)
_DEFAULT_NODE_START = 10000


@dataclass
class StagedNetwork:
    """In-memory network value staged for commit to the project tables.

    :Attributes:
        **nodes** (:obj:`gpd.GeoDataFrame`): At minimum ``node_id`` (int),
        ``geometry`` (Point, EPSG:4326), ``modes`` (str). May carry arbitrary
        free-form columns and columns starting with ``_`` (scratch columns).

        **links** (:obj:`gpd.GeoDataFrame`): At minimum ``link_id``, ``a_node``,
        ``b_node``, ``direction``, ``modes``, ``link_type``, ``distance`` (m),
        ``geometry`` (LineString, EPSG:4326). May carry arbitrary free-form
        columns. ``_source_id`` is the scratch column used by the simplifier
        for per-merged-edge attribute reconciliation.

        **crs_geo** (:obj:`str`): Always ``"EPSG:4326"`` at staged-network
        boundaries.

        **source_meta** (:obj:`dict`): Drives both ``other_attributes``
        per-row provenance and the ``about``-table whole-import provenance.
        Keys: ``source``, ``backend``, ``source_url``, ``release``,
        ``fetched_at``, ``download_cache``.
    """

    nodes: gpd.GeoDataFrame
    links: gpd.GeoDataFrame
    crs_geo: str = "EPSG:4326"
    source_meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Assert the schema invariants of this staged network."""
        missing_nodes = [c for c in _REQUIRED_NODE_COLS if c not in self.nodes.columns]
        if missing_nodes:
            raise StagedNetworkValidationError(
                f"nodes GeoDataFrame missing required columns: {missing_nodes}"
            )
        missing_links = [c for c in _REQUIRED_LINK_COLS if c not in self.links.columns]
        if missing_links:
            raise StagedNetworkValidationError(
                f"links GeoDataFrame missing required columns: {missing_links}"
            )

        if self.nodes.crs is None or str(self.nodes.crs).upper() not in (
            "EPSG:4326",
            str(self.crs_geo).upper(),
        ):
            raise StagedNetworkValidationError(
                f"nodes CRS must be EPSG:4326, got {self.nodes.crs}"
            )
        if self.links.crs is None or str(self.links.crs).upper() not in (
            "EPSG:4326",
            str(self.crs_geo).upper(),
        ):
            raise StagedNetworkValidationError(
                f"links CRS must be EPSG:4326, got {self.links.crs}"
            )

        if not np.issubdtype(self.nodes["node_id"].dtype, np.integer):
            raise StagedNetworkValidationError(
                f"nodes.node_id must be integer dtype, got {self.nodes['node_id'].dtype}"
            )
        if self.nodes["node_id"].duplicated().any():
            raise StagedNetworkValidationError("nodes.node_id contains duplicates")
        if (self.nodes["node_id"] < _DEFAULT_NODE_START).any():
            raise StagedNetworkValidationError(
                f"nodes.node_id values must be >= {_DEFAULT_NODE_START}"
            )

        node_ids = set(self.nodes["node_id"].tolist())
        a_missing = ~self.links["a_node"].isin(node_ids)
        b_missing = ~self.links["b_node"].isin(node_ids)
        if a_missing.any():
            raise StagedNetworkValidationError(
                f"{int(a_missing.sum())} links.a_node values are not in nodes.node_id"
            )
        if b_missing.any():
            raise StagedNetworkValidationError(
                f"{int(b_missing.sum())} links.b_node values are not in nodes.node_id"
            )

        if (self.links["distance"] <= 0).any():
            raise StagedNetworkValidationError("links.distance must be > 0 (metres)")

        if not self.links["direction"].isin([-1, 0, 1]).all():
            raise StagedNetworkValidationError("links.direction values must be in {-1, 0, 1}")

        if (self.links["modes"].fillna("").str.len() == 0).any():
            raise StagedNetworkValidationError(
                "links.modes must be a non-empty string for every row"
            )

    def to_graph(self) -> "nx.MultiDiGraph":
        """Build a networkx MultiDiGraph copy of the staged network.

        Used as the canonical input shape for the OSMnx simplifier. Free-form
        columns flow through as edge / node attributes.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        g.graph["crs"] = self.crs_geo

        for _, row in self.nodes.iterrows():
            attrs = {k: v for k, v in row.items() if k != "geometry"}
            geom = row.geometry
            attrs["x"] = geom.x
            attrs["y"] = geom.y
            g.add_node(int(row["node_id"]), **attrs)

        for _, row in self.links.iterrows():
            attrs = {k: v for k, v in row.items() if k != "geometry"}
            attrs["geometry"] = row.geometry
            a = int(row["a_node"])
            b = int(row["b_node"])
            direction = int(row["direction"])
            if direction == 1:
                g.add_edge(a, b, key=int(row["link_id"]), **attrs)
            elif direction == -1:
                g.add_edge(b, a, key=int(row["link_id"]), **attrs)
            else:
                # bidirectional → two directed edges
                g.add_edge(a, b, key=int(row["link_id"]), **attrs)
                g.add_edge(b, a, key=int(row["link_id"]), **attrs)
        return g

    @classmethod
    def from_graph(cls, g, source_meta=None):
        """Convert a MultiDiGraph (as produced by OSMnx) back to a StagedNetwork."""
        from shapely.geometry import Point

        node_records = []
        for nid, data in g.nodes(data=True):
            geom = data.get("geometry")
            if geom is None:
                geom = Point(data.get("x"), data.get("y"))
            rec = dict(data)
            rec["node_id"] = int(nid)
            rec["geometry"] = geom
            node_records.append(rec)

        link_records = []
        for u, v, k, data in g.edges(keys=True, data=True):
            rec = dict(data)
            rec.setdefault("a_node", int(u))
            rec.setdefault("b_node", int(v))
            rec.setdefault("link_id", int(k) if isinstance(k, int) else 0)
            link_records.append(rec)

        crs = g.graph.get("crs", "EPSG:4326")
        nodes_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs=crs)
        links_gdf = gpd.GeoDataFrame(link_records, geometry="geometry", crs=crs)
        return cls(
            nodes=nodes_gdf, links=links_gdf, crs_geo=str(crs), source_meta=source_meta or {}
        )
