"""Internal helpers for preparing routing inputs and mapping local IDs."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

from aequilibrae.paths.cython.context import GraphContext, NodeBasedContext, TurnBasedContext

if TYPE_CHECKING:
    from aequilibrae.paths.graph import Graph


def _copy_ids(values: np.ndarray, count: int, name: str) -> np.ndarray:
    values = np.asarray(values)

    if values.shape != (count,):
        raise ValueError(f"{name} must have {count} entries")

    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")

    values = np.array(values, order="C", copy=True)
    values.flags.writeable = False

    return values


class GraphMapping:
    """Map a routing context's local indices to copied external IDs.

    Node IDs follow context node order. Link IDs and directions follow context
    link order. This object does not read Graph or expand compressed links.
    """

    def __init__(
        self,
        context: GraphContext,
        node_ids: np.ndarray,
        link_ids: np.ndarray,
        directions: np.ndarray,
    ):
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a routing context")

        self.context = context
        self.node_ids = _copy_ids(node_ids, context.node_count, "node_ids")
        self.link_ids = _copy_ids(link_ids, context.link_count, "link_ids")
        self.directions = _copy_ids(directions, context.link_count, "directions")

        if not np.all(np.isin(self.directions, [-1, 1])):
            raise ValueError("directions must be -1 or 1")

        self._node_indices = {node: index for index, node in enumerate(self.node_ids)}

        if len(self._node_indices) != context.node_count:
            raise ValueError("node_ids must be unique")

    def node_index(self, node_id: int) -> int:
        """Find the local index of an external node ID."""
        if isinstance(node_id, (bool, np.bool_)):
            raise TypeError("node_id must be an integer, not a boolean")

        node_id = operator.index(node_id)

        try:
            return self._node_indices[node_id]
        except KeyError:
            raise ValueError(f"Node {node_id} is not present in the routing context") from None

    def path_nodes(self, origin: int, links: np.ndarray) -> np.ndarray:
        """Map a local origin and path links to external node IDs.

        Links must be in path order, as returned by SearchResults.path_links_to.
        An empty path returns only the origin. Check reachability before calling.
        """
        nodes = np.empty(len(links) + 1, dtype=self.node_ids.dtype)
        nodes[0] = self.node_ids[origin]
        nodes[1:] = self.node_ids[self.context.heads[links]]

        return nodes


def make_routing_context(graph: Graph, costs: np.ndarray | None = None, *, compact: bool = False) -> GraphContext:
    """Copy topology from Graph into a full or compact routing context.

    Without costs, copy the graph's current cost buffer too. Supplied costs are
    borrowed so assignment can update its own buffer between searches. No Graph
    arrays are changed or retained by the context when costs are omitted.
    """
    if compact:
        links = graph.compact_graph
        offsets = graph.compact_fs
        turn_offsets = graph.compact_turn_fs
        turn_links = graph.compact_turn_to_arcs
        turn_penalties = graph.compact_turn_penalties
    else:
        links = graph.graph
        offsets = graph.fs
        turn_offsets = graph.turn_fs
        turn_links = graph.turn_to_arcs
        turn_penalties = graph.turn_penalties

    link_count = len(links)

    if costs is None:
        if not graph.cost_field:
            raise ValueError("Set the graph cost field before preparing routing")

        source = graph.compact_cost if compact else graph.cost
        costs = np.array(source[:link_count], dtype=np.float64, order="C", copy=True)
        costs.flags.writeable = False

    topology = (offsets, links.b_node.to_numpy(), costs)

    if graph.has_turn_restrictions:
        # Keep Graph's connector bans rather than also blocking centroid nodes.
        # Compact routing uses Graph's turn tables without rebuilding them.
        return TurnBasedContext(
            *topology,
            turn_fs=turn_offsets,
            turn_to_links=turn_links,
            turn_penalties=turn_penalties,
            allow_uturns=graph.allow_path_uturns,
            blocked_centroid_count=0,
        )

    return NodeBasedContext(
        *topology,
        blocked_centroid_count=graph.num_zones if graph.block_centroid_flows else 0,
    )
