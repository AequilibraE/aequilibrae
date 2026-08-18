"""Composable, human-scale network definitions and independent shortest-path oracles."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from aequilibrae.paths import Graph

if TYPE_CHECKING:
    from aequilibrae.paths.results import PathResults


@dataclass(frozen=True)
class NodeDef:
    """A named node. Coordinates are source data; geometry is derived from them."""

    name: str
    x: float
    y: float
    centroid: bool = False


@dataclass(frozen=True)
class LinkDef:
    """A named physical link using local node names."""

    name: str
    a: str
    b: str
    cost: float
    direction: Literal[-1, 0, 1]

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError(f"Link {self.name!r} has invalid direction {self.direction}")
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError(f"Link {self.name!r} must have a finite non-negative cost")


@dataclass(frozen=True)
class TurnDef:
    """A movement control using local node names; non-finite penalties prohibit."""

    from_node: str
    via_node: str
    to_node: str
    penalty: float | None


@dataclass(frozen=True)
class BridgeDef:
    """A physical link between qualified ``component:node`` names."""

    name: str
    a: str
    b: str
    cost: float = 1.0
    direction: Literal[-1, 0, 1] = 0

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError(f"Bridge {self.name!r} has invalid direction {self.direction}")
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError(f"Bridge {self.name!r} must have a finite non-negative cost")


@dataclass(frozen=True)
class NetworkComponent:
    """One independently understandable pathological network component."""

    name: str
    purpose: str
    nodes: tuple[NodeDef, ...]
    links: tuple[LinkDef, ...]
    turns: tuple[TurnDef, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or ":" in self.name:
            raise ValueError("Component names must be non-empty and cannot contain ':'")
        if not self.purpose.strip():
            raise ValueError(f"Component {self.name!r} needs a human-readable purpose")

        node_names = [node.name for node in self.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError(f"Component {self.name!r} has duplicate node names")
        link_names = [link.name for link in self.links]
        if len(link_names) != len(set(link_names)):
            raise ValueError(f"Component {self.name!r} has duplicate link names")

        known_nodes = set(node_names)
        for link in self.links:
            if link.a not in known_nodes or link.b not in known_nodes:
                raise ValueError(f"Link {self.name}:{link.name} references an unknown node")
            if link.a == link.b:
                raise ValueError(f"Link {self.name}:{link.name} is a self-loop; use a two-arc U-turn instead")
        for turn in self.turns:
            movement_nodes = {turn.from_node, turn.via_node, turn.to_node}
            if not movement_nodes.issubset(known_nodes):
                raise ValueError(f"Turn {turn!r} in {self.name!r} references an unknown node")


@dataclass(frozen=True)
class DirectedArc:
    """One traversable direction of a physical link."""

    link_id: int
    direction: Literal[-1, 1]
    tail: int
    head: int
    cost: float

    @property
    def directed_link(self) -> tuple[int, int]:
        return self.link_id, self.direction


@dataclass(frozen=True)
class OraclePath:
    """A NetworkX arc-state shortest path with its independent cost breakdown."""

    nodes: tuple[int, ...]
    directed_links: tuple[tuple[int, int], ...]
    cost: float
    link_cost: float
    turn_cost: float


@dataclass(frozen=True)
class PathologicalNetwork:
    """A composition that assigns stable IDs and computes all geometries on demand."""

    components: tuple[NetworkComponent, ...]
    bridges: tuple[BridgeDef, ...] = ()
    component_gap: float = 4.0
    crs: str = "EPSG:3857"

    def __post_init__(self) -> None:
        component_names = [component.name for component in self.components]
        if not component_names:
            raise ValueError("A pathological network needs at least one component")
        if len(component_names) != len(set(component_names)):
            raise ValueError("Component names must be unique when composed")

        known_nodes = set(self._node_keys())
        bridge_names: set[str] = set()
        for bridge in self.bridges:
            if bridge.name in bridge_names:
                raise ValueError(f"Duplicate bridge name {bridge.name!r}")
            bridge_names.add(bridge.name)
            if bridge.a not in known_nodes or bridge.b not in known_nodes:
                raise ValueError(f"Bridge {bridge.name!r} references an unknown qualified node")

    @classmethod
    def compose(
        cls,
        *components: NetworkComponent,
        bridges: tuple[BridgeDef, ...] = (),
        component_gap: float = 4.0,
    ) -> PathologicalNetwork:
        return cls(tuple(components), bridges, component_gap)

    @staticmethod
    def _qualified(component: NetworkComponent, local_name: str) -> str:
        return f"{component.name}:{local_name}"

    def _node_keys(self) -> tuple[str, ...]:
        return tuple(self._qualified(component, node.name) for component in self.components for node in component.nodes)

    def _node_id_map(self) -> dict[str, int]:
        return {key: node_id for node_id, key in enumerate(self._node_keys(), start=1)}

    def _component_offsets(self) -> dict[str, float]:
        offsets: dict[str, float] = {}
        cursor = 0.0
        for component in self.components:
            xs = [float(node.x) for node in component.nodes]
            minimum = min(xs)
            maximum = max(xs)
            offsets[component.name] = cursor - minimum
            cursor += maximum - minimum + self.component_gap
        return offsets

    def _coordinate_map(self) -> dict[str, tuple[float, float]]:
        offsets = self._component_offsets()
        return {
            self._qualified(component, node.name): (float(node.x) + offsets[component.name], float(node.y))
            for component in self.components
            for node in component.nodes
        }

    def node_id(self, qualified_name: str) -> int:
        """Returns the stable positive ID for ``component:node``."""
        try:
            return self._node_id_map()[qualified_name]
        except KeyError as exc:
            raise KeyError(f"Unknown node {qualified_name!r}") from exc

    def link_id(self, qualified_name: str) -> int:
        """Returns the stable ID for ``component:link`` or ``bridge:name``."""
        names = [self._qualified(component, link.name) for component in self.components for link in component.links] + [
            f"bridge:{bridge.name}" for bridge in self.bridges
        ]
        try:
            return names.index(qualified_name) + 1
        except ValueError as exc:
            raise KeyError(f"Unknown link {qualified_name!r}") from exc

    @property
    def purposes(self) -> dict[str, str]:
        return {component.name: component.purpose for component in self.components}

    @property
    def centroids(self) -> np.ndarray:
        ids = self._node_id_map()
        return np.asarray(
            [
                ids[self._qualified(component, node.name)]
                for component in self.components
                for node in component.nodes
                if node.centroid
            ],
            dtype=np.int64,
        )

    def node_frame(self) -> gpd.GeoDataFrame:
        """Computes a fresh point GeoDataFrame from the coordinate definitions."""
        ids = self._node_id_map()
        coordinates = self._coordinate_map()
        rows = []
        for component in self.components:
            for node in component.nodes:
                key = self._qualified(component, node.name)
                x, y = coordinates[key]
                rows.append(
                    {
                        "node_id": ids[key],
                        "name": key,
                        "component": component.name,
                        "centroid": node.centroid,
                        "geometry": Point(x, y),
                    }
                )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.crs)

    def link_frame(self) -> gpd.GeoDataFrame:
        """Computes a fresh line GeoDataFrame from link endpoints."""
        ids = self._node_id_map()
        coordinates = self._coordinate_map()
        rows = []
        link_id = 1
        for component in self.components:
            for link in component.links:
                a_key = self._qualified(component, link.a)
                b_key = self._qualified(component, link.b)
                geometry = LineString((coordinates[a_key], coordinates[b_key]))
                rows.append(
                    {
                        "link_id": link_id,
                        "name": self._qualified(component, link.name),
                        "component": component.name,
                        "a_node": ids[a_key],
                        "b_node": ids[b_key],
                        "direction": link.direction,
                        "cost": float(link.cost),
                        "distance": float(geometry.length),
                        "geometry": geometry,
                    }
                )
                link_id += 1

        for bridge in self.bridges:
            geometry = LineString((coordinates[bridge.a], coordinates[bridge.b]))
            rows.append(
                {
                    "link_id": link_id,
                    "name": f"bridge:{bridge.name}",
                    "component": "bridge",
                    "a_node": ids[bridge.a],
                    "b_node": ids[bridge.b],
                    "direction": bridge.direction,
                    "cost": float(bridge.cost),
                    "distance": float(geometry.length),
                    "geometry": geometry,
                }
            )
            link_id += 1
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.crs)

    def turn_frame(self) -> gpd.GeoDataFrame:
        """Computes turn IDs and three-point line geometry from movement definitions."""
        ids = self._node_id_map()
        coordinates = self._coordinate_map()
        rows = []
        for component in self.components:
            for turn_id, turn in enumerate(component.turns, start=len(rows) + 1):
                from_key = self._qualified(component, turn.from_node)
                via_key = self._qualified(component, turn.via_node)
                to_key = self._qualified(component, turn.to_node)
                rows.append(
                    {
                        "turn_id": turn_id,
                        "component": component.name,
                        "from_node": ids[from_key],
                        "via_node": ids[via_key],
                        "to_node": ids[to_key],
                        "penalty": turn.penalty,
                        "geometry": LineString((coordinates[from_key], coordinates[via_key], coordinates[to_key])),
                    }
                )
        columns = ["turn_id", "component", "from_node", "via_node", "to_node", "penalty", "geometry"]
        return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=self.crs)

    def directed_arcs(self) -> tuple[DirectedArc, ...]:
        arcs: list[DirectedArc] = []
        for row in self.link_frame().itertuples(index=False):
            if row.direction >= 0:
                arcs.append(DirectedArc(int(row.link_id), 1, int(row.a_node), int(row.b_node), float(row.cost)))
            if row.direction <= 0:
                arcs.append(DirectedArc(int(row.link_id), -1, int(row.b_node), int(row.a_node), float(row.cost)))
        return tuple(arcs)

    def networkx_graph(self) -> nx.MultiDiGraph:
        """Builds the ordinary directed NetworkX graph, without movement controls."""
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self._node_id_map().values())
        for arc in self.directed_arcs():
            graph.add_edge(
                arc.tail,
                arc.head,
                key=arc.directed_link,
                weight=arc.cost,
                link_id=arc.link_id,
                direction=arc.direction,
            )
        return graph

    @staticmethod
    def _normalise_penalty(value: object) -> float:
        if value is None or pd.isna(value):
            return inf
        penalty = float(value)
        if penalty < 0:
            raise ValueError("Turn penalties must be non-negative")
        return penalty if isfinite(penalty) else inf

    def turn_lookup(self) -> dict[tuple[int, int, int], float]:
        """Applies the suite's explicit duplicate precedence contract."""
        lookup: dict[tuple[int, int, int], float] = {}
        for row in self.turn_frame().itertuples(index=False):
            key = (int(row.from_node), int(row.via_node), int(row.to_node))
            penalty = self._normalise_penalty(row.penalty)
            if key not in lookup or isfinite(lookup[key]):
                lookup[key] = penalty
        return lookup

    def oracle_state_graph(
        self,
        origin: int,
        destination: int,
        *,
        allow_path_uturns: bool = False,
        block_centroid_flows: bool = False,
    ) -> nx.DiGraph:
        """Builds the independent arc-state graph used as the shortest-path oracle."""
        graph = nx.DiGraph()
        source = ("source", origin)
        sink = ("sink", destination)
        graph.add_nodes_from((source, sink))

        arcs = self.directed_arcs()
        outgoing: dict[int, list[DirectedArc]] = {}
        incoming: dict[int, list[DirectedArc]] = {}
        for arc in arcs:
            outgoing.setdefault(arc.tail, []).append(arc)
            incoming.setdefault(arc.head, []).append(arc)
            if arc.tail == origin:
                graph.add_edge(source, arc, weight=arc.cost)
            if arc.head == destination:
                graph.add_edge(arc, sink, weight=0.0)

        restrictions = self.turn_lookup()
        centroids = {int(node) for node in self.centroids}
        for via_node in set(incoming) & set(outgoing):
            for from_arc in incoming[via_node]:
                for to_arc in outgoing[via_node]:
                    is_uturn = to_arc.head == from_arc.tail
                    if is_uturn and not allow_path_uturns:
                        continue
                    if block_centroid_flows and via_node in centroids:
                        continue
                    penalty = restrictions.get((from_arc.tail, via_node, to_arc.head), 0.0)
                    if not isfinite(penalty):
                        continue
                    graph.add_edge(from_arc, to_arc, weight=to_arc.cost + penalty, turn_penalty=penalty)
        return graph

    def oracle_path(
        self,
        origin: int,
        destination: int,
        *,
        allow_path_uturns: bool = False,
        block_centroid_flows: bool = False,
    ) -> OraclePath:
        """Computes the movement-aware shortest path with NetworkX Dijkstra."""
        if origin == destination:
            return OraclePath((origin,), (), 0.0, 0.0, 0.0)

        graph = self.oracle_state_graph(
            origin,
            destination,
            allow_path_uturns=allow_path_uturns,
            block_centroid_flows=block_centroid_flows,
        )
        states = nx.shortest_path(graph, ("source", origin), ("sink", destination), weight="weight")
        arcs = tuple(state for state in states if isinstance(state, DirectedArc))
        link_cost = sum(arc.cost for arc in arcs)
        lookup = self.turn_lookup()
        turn_cost = sum(
            lookup.get((from_arc.tail, from_arc.head, to_arc.head), 0.0)
            for from_arc, to_arc in zip(arcs[:-1], arcs[1:], strict=True)
        )
        nodes = (arcs[0].tail,) + tuple(arc.head for arc in arcs)
        return OraclePath(
            nodes=nodes,
            directed_links=tuple(arc.directed_link for arc in arcs),
            cost=float(link_cost + turn_cost),
            link_cost=float(link_cost),
            turn_cost=float(turn_cost),
        )

    def build_graph(
        self,
        *,
        restriction_timing: Literal["before_prepare", "after_prepare", "none"] = "after_prepare",
        remove_dead_ends: bool = False,
        allow_uturns_everywhere: bool = False,
        allow_path_uturns: bool = False,
        block_centroid_flows: bool = False,
    ) -> Graph:
        """Builds an AequilibraE graph exclusively through public lifecycle methods."""
        graph = Graph()
        graph.network = pd.DataFrame(self.link_frame().drop(columns="geometry"))
        graph.mode = "c"
        turns = pd.DataFrame(self.turn_frame().drop(columns="geometry"))

        if restriction_timing == "before_prepare":
            graph.set_turn_restrictions(turns, allow_path_uturns=allow_path_uturns)
        elif restriction_timing not in {"after_prepare", "none"}:
            raise ValueError(f"Unknown restriction timing {restriction_timing!r}")

        centroids = self.centroids if self.centroids.size else None
        graph.prepare_graph(
            centroids,
            remove_dead_ends=remove_dead_ends,
            allow_uturns_everywhere=allow_uturns_everywhere,
        )
        graph.set_graph("cost")
        if centroids is not None:
            graph.set_blocked_centroid_flows(block_centroid_flows)

        if restriction_timing == "after_prepare":
            graph.set_turn_restrictions(turns, allow_path_uturns=allow_path_uturns)
        return graph

    @staticmethod
    def result_directed_links(result: PathResults) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(link_id), int(direction))
            for link_id, direction in zip(result.path, result.path_link_directions, strict=True)
        )

    def result_generalized_cost(
        self,
        result: PathResults,
        *,
        allow_path_uturns: bool = False,
        block_centroid_flows: bool = False,
    ) -> float:
        """Reprices an AequilibraE path from definitions, never from its reported cost."""
        arc_lookup = {arc.directed_link: arc for arc in self.directed_arcs()}
        arcs = tuple(arc_lookup[item] for item in self.result_directed_links(result))
        turn_lookup = self.turn_lookup()
        centroids = {int(node) for node in self.centroids}
        cost = sum(arc.cost for arc in arcs)
        for from_arc, to_arc in zip(arcs[:-1], arcs[1:], strict=True):
            if to_arc.head == from_arc.tail and not allow_path_uturns:
                return inf
            if block_centroid_flows and from_arc.head in centroids:
                return inf
            cost += turn_lookup.get((from_arc.tail, from_arc.head, to_arc.head), 0.0)
        return float(cost)

    def dump_geoparquet(self, directory: Path) -> dict[str, Path]:
        """Dumps fresh geometries for visual inspection in QGIS."""
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "nodes": directory / "nodes.parquet",
            "links": directory / "links.parquet",
            "turns": directory / "turns.parquet",
        }
        self.node_frame().to_parquet(paths["nodes"], index=False)
        self.link_frame().to_parquet(paths["links"], index=False)
        self.turn_frame().to_parquet(paths["turns"], index=False)
        return paths
