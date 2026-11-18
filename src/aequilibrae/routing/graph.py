from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aequilibrae.routing.connectivity_analysis import disconnected_analysis
from aequilibrae.routing.cython.context import GraphContext, SkimmingContext, TurnBasedContext
from aequilibrae.routing.mapping import GraphMapping

if TYPE_CHECKING:
    from aequilibrae.routing.network_skimming import NetworkSkimming
    from aequilibrae.routing.results.path_results import PathResults


@dataclasses.dataclass(frozen=True, eq=False)
class Graph:
    """One routing context with fixed topology and mappings to source links.

    Create graphs with ``Network.build_graph()``.
    Network tables and construction intermediates are not retained. Routing,
    correspondence and skim fields belong to their respective sub-objects.
    Reconstruct the graph to change its configuration; routing costs can still
    be rebound independently of the fixed topology.
    """

    routing: GraphContext
    mapping: GraphMapping
    skimming: SkimmingContext
    mode: str
    cost_field: str | None = None

    def __repr__(self) -> str:
        return (
            f"Graph(nodes={self.num_nodes}, links={self.num_links}, zones={self.num_zones}, "
            f"mode={self.mode!r}, cost_field={self.cost_field!r}, skim_fields={self.skim_fields!r}, "
            f"compact={self.compact}, remove_dead_ends={self.remove_dead_ends}, "
            f"allow_uturns={self.allow_uturns}, block_centroid_flows={self.block_centroid_flows}, "
            f"turn_restrictions={self.has_turn_restrictions})"
        )

    def __str__(self) -> str:
        return (
            "Graph\n"
            f"  Nodes: {self.num_nodes}; links: {self.num_links}; zones: {self.num_zones}\n"
            f"  Mode: {self.mode}, cost: {self.cost_field!r}, skim fields: {self.skim_fields!r}\n"
            f"  Compact: {self.compact}, remove dead ends: {self.remove_dead_ends}, "
            f"allow U-turns: {self.allow_uturns}\n"
            f"  Block centroid flows: {self.block_centroid_flows}, "
            f"turn restrictions: {self.has_turn_restrictions}"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.routing, GraphContext) or not isinstance(self.mapping, GraphMapping):
            raise ValueError("Graph requires a routing context and a GraphMapping")
        if not isinstance(self.skimming, SkimmingContext):
            raise ValueError("skimming must be a SkimmingContext")
        if self.routing.node_count != self.mapping.node_count or self.routing.link_count != self.mapping.link_count:
            raise ValueError("Routing context and mapping dimensions must agree")
        if self.skimming.link_count != self.routing.link_count:
            raise ValueError("Skimming and routing contexts must have the same link count")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a nonempty string")
        if self.cost_field is not None and (not isinstance(self.cost_field, str) or not self.cost_field):
            raise ValueError("cost_field must be a nonempty string")

    @property
    def compact(self) -> bool:
        """Whether link contraction was requested when building the mapping."""
        return self.mapping.compact

    @property
    def remove_dead_ends(self) -> bool:
        """Whether dead-end removal was requested when building the mapping."""
        return self.mapping.remove_dead_ends

    @property
    def allow_uturns(self) -> bool:
        """Whether U-turns were allowed when building the routing context."""
        return self.routing.allow_uturns

    @property
    def centroid_count(self) -> int:
        """Number of centroids supplied when building the routing context."""
        return self.routing.centroid_count

    @property
    def block_centroid_flows(self) -> bool:
        """Whether the build requested blocking travel through centroids."""
        return self.routing.block_centroid_flows

    @property
    def centroids(self) -> npt.NDArray[np.uint64]:
        return self.mapping.network_node(np.arange(self.centroid_count, dtype=np.intp))

    @property
    def all_nodes(self) -> npt.NDArray[np.uint64]:
        return self.mapping.network_node(np.arange(self.mapping.node_count, dtype=np.intp))

    @property
    def num_nodes(self) -> int:
        return self.routing.node_count

    @property
    def num_links(self) -> int:
        return self.routing.link_count

    @property
    def num_zones(self) -> int:
        return self.centroid_count

    @property
    def cost(self) -> npt.NDArray[np.float64]:
        return self.routing.costs

    @property
    def has_turn_restrictions(self) -> bool:
        return isinstance(self.routing, TurnBasedContext)

    @property
    def skim_fields(self) -> list[str]:
        return list(self.skimming.field_names)

    def set_costs(self, costs: npt.ArrayLike) -> None:
        """Replace directed network costs. Positive infinity closes a link."""
        costs = np.asarray(costs)
        if costs.shape != (self.mapping.network.link_count,):
            raise ValueError("Costs must have one value per directed network link")
        if costs.dtype.kind not in "iuf" or np.isnan(costs).any() or np.any(costs < 0):
            raise ValueError("Graph costs must be numeric, nonnegative and cannot contain NaN")

        values = self.mapping.aggregate_costs(costs.astype(np.float64, copy=False))
        values.flags.writeable = False
        self.routing.update_costs(values)

    def selected_links(self, selections: Mapping[str, Iterable[tuple[int, int]]]) -> dict[str, npt.NDArray[np.intp]]:
        """Resolve named source-link selections, raising for links absent from this graph."""
        result = {}
        for name, members in selections.items():
            indices: set[int] = set()
            for link, direction in members:
                if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)):
                    raise ValueError("Selected link directions must be integers: -1, 0 or +1")
                if direction not in (-1, 0, 1):
                    raise ValueError("Selected link directions must be -1, 0 or +1")
                matches = set()
                for sign in (1, -1) if direction == 0 else (direction,):
                    try:
                        matches.add(self.mapping.graph_link(link, sign))
                    except KeyError:
                        pass
                if not matches:
                    raise ValueError(f"Selected link {link} and direction {direction} is absent from the graph")
                indices.update(matches)
            result[name] = np.array(sorted(indices), dtype=np.intp)
        return result

    def compute_path(self, origin: int, destination: int, **kwargs: object) -> PathResults:
        """Compute a path with nodes and mileposts on this built graph.

        Additional arguments are passed to :class:`PathResults`, including
        explicit coordinates and a heuristic scale when using A*.
        """
        from aequilibrae.routing.results.path_results import PathResults

        return PathResults(self, origin, destination, **kwargs)

    def compute_skims(self, cores: int | None = None) -> NetworkSkimming:
        """Compute centroid skims using this graph's routing and skim contexts."""
        from aequilibrae.routing.network_skimming import NetworkSkimming

        skimmer = NetworkSkimming(self)
        if cores is not None:
            skimmer.set_cores(cores)
        skimmer.execute()
        return skimmer

    def disconnected_nodes(self) -> np.ndarray:
        """Identify nodes unable to reach and be reached from the main network.

        The largest strongly connected component selects the main network.
        Blocked centroids can be endpoints, but cannot connect other nodes
        through themselves. Individual turn restrictions are not considered.
        With no active links, only the first retained node is considered connected.

        :returns: Disconnected external node IDs in retained graph-node order.
        """
        # Extract active directed links from the routing context
        active = np.isfinite(self.cost)
        tails = np.repeat(np.arange(self.num_nodes), np.diff(self.routing.fs).astype(np.int64))
        return disconnected_analysis(
            tails[active],
            self.routing.heads[active].astype(np.int64),
            self.all_nodes,
            self.block_centroid_flows,
            self.centroids,
        )
