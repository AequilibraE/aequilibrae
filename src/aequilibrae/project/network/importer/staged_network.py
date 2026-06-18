from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np

from aequilibrae.project.network.importer.exceptions import StagedNetworkValidationError

if TYPE_CHECKING:
    import networkx as nx

_REQUIRED_NODE_COLS = ("node_id", "geometry", "modes")
_REQUIRED_LINK_COLS = ("link_id", "a_node", "b_node", "direction", "modes", "link_type", "distance", "geometry")
_DEFAULT_NODE_START = 10000


@dataclass
class StagedNetwork:
    nodes: gpd.GeoDataFrame
    links: gpd.GeoDataFrame
    crs_geo: str = "EPSG:4326"
    source_meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        missing_nodes = [c for c in _REQUIRED_NODE_COLS if c not in self.nodes.columns]
        if missing_nodes:
            raise StagedNetworkValidationError(f"nodes GeoDataFrame missing required columns: {missing_nodes}")
        missing_links = [c for c in _REQUIRED_LINK_COLS if c not in self.links.columns]
        if missing_links:
            raise StagedNetworkValidationError(f"links GeoDataFrame missing required columns: {missing_links}")

        allowed_crs = ("EPSG:4326", str(self.crs_geo).upper())
        if self.nodes.crs is None or str(self.nodes.crs).upper() not in allowed_crs:
            raise StagedNetworkValidationError(f"nodes CRS must be EPSG:4326, got {self.nodes.crs}")
        if self.links.crs is None or str(self.links.crs).upper() not in allowed_crs:
            raise StagedNetworkValidationError(f"links CRS must be EPSG:4326, got {self.links.crs}")

        if not np.issubdtype(self.nodes["node_id"].dtype, np.integer):
            raise StagedNetworkValidationError(f"nodes.node_id must be integer dtype, got {self.nodes['node_id'].dtype}")
        if self.nodes["node_id"].duplicated().any():
            raise StagedNetworkValidationError("nodes.node_id contains duplicates")
        if (self.nodes["node_id"] < _DEFAULT_NODE_START).any():
            raise StagedNetworkValidationError(f"nodes.node_id values must be >= {_DEFAULT_NODE_START}")

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

    def to_graph(self) -> "nx.MultiDiGraph":
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = self.crs_geo

        node_cols = [c for c in self.nodes.columns if c != "geometry"]
        xs = self.nodes.geometry.x.to_numpy()
        ys = self.nodes.geometry.y.to_numpy()
        for rec, x, y in zip(self.nodes[node_cols].to_dict(orient="records"), xs, ys):
            nid = int(rec["node_id"])
            graph.add_node(nid, x=x, y=y, **rec)

        for rec, geom in zip(
            self.links.drop(columns=["geometry"]).to_dict(orient="records"),
            self.links.geometry,
        ):
            a, b = int(rec["a_node"]), int(rec["b_node"])
            link_id = int(rec["link_id"])
            attrs = {**rec, "geometry": geom}
            direction = int(rec["direction"])
            if direction == 1:
                graph.add_edge(a, b, key=link_id, **attrs)
            elif direction == -1:
                graph.add_edge(b, a, key=link_id, **attrs)
            else:
                graph.add_edge(a, b, key=link_id, **attrs)
                graph.add_edge(b, a, key=link_id, **attrs)
        return graph

    @classmethod
    def from_graph(cls, graph, source_meta=None):
        from shapely.geometry import Point

        node_records = []
        for nid, data in graph.nodes(data=True):
            geom = data.get("geometry")
            if geom is None:
                geom = Point(data.get("x"), data.get("y"))
            rec = dict(data)
            rec["node_id"] = int(nid)
            rec["geometry"] = geom
            node_records.append(rec)

        link_records = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            rec = dict(data)
            rec.setdefault("a_node", int(u))
            rec.setdefault("b_node", int(v))
            rec.setdefault("link_id", int(key) if isinstance(key, int) else 0)
            link_records.append(rec)

        crs = graph.graph.get("crs", "EPSG:4326")
        nodes_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs=crs)
        links_gdf = gpd.GeoDataFrame(link_records, geometry="geometry", crs=crs)
        return cls(nodes=nodes_gdf, links=links_gdf, crs_geo=str(crs), source_meta=source_meta or {})
