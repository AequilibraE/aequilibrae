import geopandas as gpd
import numpy as np
from dataclasses import dataclass, field
from shapely.geometry import LineString, Point
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

        # Links must be simple LineStrings. ``to_graph`` reverses geometries via
        # ``geom.coords[::-1]``, which raises on MultiLineString, so reject mixed
        # geometry up front with a clear message instead of an opaque crash.
        link_types = set(self.links.geometry.geom_type.dropna().unique())
        if link_types - {"LineString"}:
            raise StagedNetworkValidationError(
                f"links.geometry must contain only LineString geometries, found: {sorted(link_types)}"
            )

    def to_graph(self) -> "nx.MultiDiGraph":
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = self.crs_geo

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
                "_travel_speed": rec.get("speed_ab"),
                "_travel_lanes": rec.get("lanes_ab"),
            }
            if direction == 1:
                graph.add_edge(a, b, key=link_id, **attrs_ab)
                continue

            rev_geom = LineString(geom.coords[::-1]) if geom is not None else None
            attrs_ba = {
                **rec,
                "geometry": rev_geom,
                "_source_ref": f"{base_source_id}::ba",
                "_travel_speed": rec.get("speed_ba"),
                "_travel_lanes": rec.get("lanes_ba"),
            }
            if direction == -1:
                graph.add_edge(b, a, key=link_id, **attrs_ba)
            else:
                graph.add_edge(a, b, key=link_id, **attrs_ab)
                graph.add_edge(b, a, key=link_id, **attrs_ba)
        return graph

    @classmethod
    def from_graph(cls, graph, source_meta=None):
        """Reconstruct a canonical staged network from a directed MultiDiGraph.

        ``to_graph`` decomposes every bidirectional link into a pair of directed
        edges tagged ``<base>::ab`` / ``<base>::ba`` via ``_source_ref``. This
        method recombines those pairs back into a single undirected staged link
        (recovering ``direction`` and the directional speed/lane fields) so the
        round-trip preserves link cardinality and never emits duplicate
        ``link_id`` values.
        """
        node_records = []
        for nid, data in graph.nodes(data=True):
            geom = data.get("geometry")
            if geom is None:
                geom = Point(data.get("x"), data.get("y"))
            rec = dict(data)
            # ``x``/``y`` are graph-only scratch attributes that to_graph()
            # re-derives from geometry; keeping them would collide on the next
            # to_graph() call (add_node(..., x=.., **rec)).
            rec.pop("x", None)
            rec.pop("y", None)
            rec["node_id"] = int(nid)
            rec["geometry"] = geom
            node_records.append(rec)

        directed = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            source_ref = data.get("_source_ref")
            if source_ref is None:
                raise StagedNetworkValidationError(
                    "from_graph() requires every edge to carry a '_source_ref' produced by to_graph(); "
                    "one or more edges are missing it, so direction cannot be reconstructed safely"
                )
            base, _, suffix = str(source_ref).partition("::")
            directed.append((base, suffix, int(u), int(v), key, data))

        link_records = _canonicalize_directed_edges(directed)

        crs = graph.graph.get("crs", "EPSG:4326")
        nodes_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs=crs)
        links_gdf = gpd.GeoDataFrame(link_records, geometry="geometry", crs=crs)
        return cls(nodes=nodes_gdf, links=links_gdf, crs_geo=str(crs), source_meta=source_meta or {})


def _base_source_id(rec: dict) -> str:
    source_id = rec.get("source_id")
    if source_id is None:
        return str(rec["link_id"])
    return str(source_id)


_SCRATCH_KEYS = ("_source_ref", "_travel_speed", "_travel_lanes")


def _clean_edge_data(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in _SCRATCH_KEYS}


def _canonicalize_directed_edges(directed: list) -> list:
    """Recombine directed (ab/ba) edges into one staged record per base link."""
    grouped: dict = {}
    order: list = []
    for base, suffix, u, v, _key, data in directed:
        if base not in grouped:
            grouped[base] = {}
            order.append(base)
        grouped[base][suffix] = (u, v, data)

    link_records = []
    for new_id, base in enumerate(order, start=1):
        sides = grouped[base]
        ab = sides.get("ab")
        ba = sides.get("ba")

        if ab is not None and ba is not None:
            direction = 0
            u, v, data = ab
        elif ab is not None:
            direction = 1
            u, v, data = ab
        else:
            # Only the BA side survived: the canonical AB orientation is the
            # reverse of the stored geometry/endpoints.
            v, u, data = ba
            direction = -1

        rec = _clean_edge_data(data)
        rec["a_node"] = int(u)
        rec["b_node"] = int(v)
        rec["link_id"] = int(new_id)
        rec["direction"] = int(direction)

        if direction == -1 and rec.get("geometry") is not None:
            rec["geometry"] = LineString(rec["geometry"].coords[::-1])

        ab_data = ab[2] if ab is not None else None
        ba_data = ba[2] if ba is not None else None
        if ab_data is not None and ab_data.get("_travel_speed") is not None:
            rec["speed_ab"] = ab_data.get("_travel_speed")
        if ab_data is not None and ab_data.get("_travel_lanes") is not None:
            rec["lanes_ab"] = ab_data.get("_travel_lanes")
        if ba_data is not None and ba_data.get("_travel_speed") is not None:
            rec["speed_ba"] = ba_data.get("_travel_speed")
        if ba_data is not None and ba_data.get("_travel_lanes") is not None:
            rec["lanes_ba"] = ba_data.get("_travel_lanes")

        link_records.append(rec)
    return link_records

