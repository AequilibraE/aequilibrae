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
        for label, gdf, required in (
            ("nodes", self.nodes, _REQUIRED_NODE_COLS),
            ("links", self.links, _REQUIRED_LINK_COLS),
        ):
            missing = [c for c in required if c not in gdf.columns]
            if missing:
                raise StagedNetworkValidationError(f"{label} GeoDataFrame missing required columns: {missing}")
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
        for endpoint in ("a_node", "b_node"):
            missing = ~self.links[endpoint].isin(node_ids)
            if missing.any():
                raise StagedNetworkValidationError(
                    f"{int(missing.sum())} links.{endpoint} values are not in nodes.node_id"
                )

        bad_distance = self.links["distance"] <= 0
        if bad_distance.any():
            columns = [
                column
                for column in ("link_id", "source_id", "a_node", "b_node", "distance")
                if column in self.links.columns
            ]
            bad_links = self.links.loc[bad_distance, columns]
            sample = bad_links.head(10).to_dict(orient="records")
            suffix = f" (+{len(bad_links) - 10} more)" if len(bad_links) > 10 else ""
            raise StagedNetworkValidationError(
                f"links.distance must be > 0 (metres); offending links: {sample}{suffix}"
            )
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

