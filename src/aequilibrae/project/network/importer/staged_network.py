import geopandas as gpd
import numpy as np
from dataclasses import dataclass, field
from shapely.geometry import LineString
from typing import TYPE_CHECKING

from aequilibrae.project.network.importer.exceptions import StagedNetworkValidationError
from aequilibrae.project.network.importer.utils import NODE_ID_START

if TYPE_CHECKING:
    import networkx as nx

_REQUIRED_NODE_COLS = ("node_id", "geometry", "modes")
_REQUIRED_LINK_COLS = ("link_id", "a_node", "b_node", "direction", "modes", "link_type", "distance", "geometry")


@dataclass
class StagedNetwork:
    nodes: gpd.GeoDataFrame
    links: gpd.GeoDataFrame
    source_meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        missing_nodes = [c for c in _REQUIRED_NODE_COLS if c not in self.nodes.columns]
        if missing_nodes:
            raise StagedNetworkValidationError(f"nodes GeoDataFrame missing required columns: {missing_nodes}")
        missing_links = [c for c in _REQUIRED_LINK_COLS if c not in self.links.columns]
        if missing_links:
            raise StagedNetworkValidationError(f"links GeoDataFrame missing required columns: {missing_links}")

        for label, gdf in (("nodes", self.nodes), ("links", self.links)):
            if gdf.crs is None or str(gdf.crs).upper() != "EPSG:4326":
                raise StagedNetworkValidationError(f"{label} CRS must be EPSG:4326, got {gdf.crs}")

        if not np.issubdtype(self.nodes["node_id"].dtype, np.integer):
            dtype = self.nodes["node_id"].dtype
            raise StagedNetworkValidationError(f"nodes.node_id must be integer dtype, got {dtype}")
        if self.nodes["node_id"].duplicated().any():
            raise StagedNetworkValidationError("nodes.node_id contains duplicates")
        if (self.nodes["node_id"] < NODE_ID_START).any():
            raise StagedNetworkValidationError(f"nodes.node_id values must be >= {NODE_ID_START}")

        node_ids = set(self.nodes["node_id"].tolist())
        a_missing = ~self.links["a_node"].isin(node_ids)
        b_missing = ~self.links["b_node"].isin(node_ids)
        if a_missing.any():
            raise StagedNetworkValidationError(f"{int(a_missing.sum())} links.a_node values are not in nodes.node_id")
        if b_missing.any():
            raise StagedNetworkValidationError(f"{int(b_missing.sum())} links.b_node values are not in nodes.node_id")

        if (self.links["distance"] <= 0).any():
            raise StagedNetworkValidationError("links.distance must be > 0 (metres)")
        if not self.links["direction"].isin([-1, 0, 1]).all():
            raise StagedNetworkValidationError("links.direction values must be in {-1, 0, 1}")
        if (self.links["modes"].fillna("").str.len() == 0).any():
            raise StagedNetworkValidationError("links.modes must be a non-empty string for every row")

        # ``to_graph`` reverses geometries via ``coords[::-1]``, which raises on
        # MultiLineString, so reject non-LineString geometry up front.
        link_types = set(self.links.geometry.geom_type.dropna().unique())
        if link_types - {"LineString"}:
            raise StagedNetworkValidationError(
                f"links.geometry must contain only LineString geometries, found: {sorted(link_types)}"
            )

    def to_graph(self) -> "nx.MultiDiGraph":
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "EPSG:4326"

        node_cols = [c for c in self.nodes.columns if c != "geometry"]
        xs = self.nodes.geometry.x.to_numpy()
        ys = self.nodes.geometry.y.to_numpy()
        for rec, x, y in zip(self.nodes[node_cols].to_dict(orient="records"), xs, ys, strict=True):
            nid = int(rec["node_id"])
            graph.add_node(nid, x=x, y=y, **rec)

        for rec, geom in zip(
            self.links.drop(columns=["geometry"]).to_dict(orient="records"),
            self.links.geometry,
            strict=True,
        ):
            a, b = int(rec["a_node"]), int(rec["b_node"])
            link_id = int(rec["link_id"])
            direction = int(rec["direction"])
            base_source_id = _base_source_id(rec)
            attrs_ab = {
                **rec,
                "geometry": geom,
                "_source_ref": f"{base_source_id}::ab",
            }
            if direction == 1:
                graph.add_edge(a, b, key=link_id, **attrs_ab)
                continue

            rev_geom = LineString(geom.coords[::-1]) if geom is not None else None
            attrs_ba = {
                **rec,
                "geometry": rev_geom,
                "_source_ref": f"{base_source_id}::ba",
            }
            if direction == -1:
                graph.add_edge(b, a, key=link_id, **attrs_ba)
            else:
                graph.add_edge(a, b, key=link_id, **attrs_ab)
                graph.add_edge(b, a, key=link_id, **attrs_ba)
        return graph


def _base_source_id(rec: dict) -> str:
    source_id = rec.get("source_id")
    if source_id is None:
        return str(rec["link_id"])
    return str(source_id)

