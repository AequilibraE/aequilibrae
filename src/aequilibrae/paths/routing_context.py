from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aequilibrae.paths.cython.context import GraphContext, NodeBasedContext, TurnBasedContext
from aequilibrae.paths.graph_building import normalise_turn_penalty
from aequilibrae.paths.mapping import GraphMapping as NetworkGraphMapping


def _copy_ids(values: np.ndarray, count: int | None, name: str) -> np.ndarray:
    """Copy node IDs to a C-contiguous, read-only array."""
    values = np.asarray(values)

    if count is not None and values.shape != (count,):
        raise ValueError(f"{name} must have {count} entries")

    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")

    values = np.array(values, order="C", copy=True)
    values.flags.writeable = False

    return values


class GraphMapping:
    """
    Map a routing context's local indices to copied external IDs.

    Node IDs follow context node order. Link IDs and directions follow context
    link order. This object does not read Graph or expand compressed links.
    """

    def __init__(self, context: GraphContext, node_ids: np.ndarray, link_ids: np.ndarray, directions: np.ndarray):
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
        import operator

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


def _record_turn_penalty(
    turn_penalties: dict[tuple[int, int], float],
    incoming_link: int,
    outgoing_link: int,
    penalty: float,
) -> None:
    """Store a turn penalty, unless it was already banned."""
    link_pair = (incoming_link, outgoing_link)
    if not np.isinf(turn_penalties.get(link_pair, 0.0)):
        turn_penalties[link_pair] = penalty


def _map_explicit_turn_penalties(
    mapping: NetworkGraphMapping,
    turn_restrictions: pd.DataFrame,
    first_network_links: NDArray[np.intp],
    last_network_links: NDArray[np.intp],
) -> dict[tuple[int, int], float]:
    """Map node-sequence restrictions to directed arc pairs."""
    network = mapping.network
    incoming_links_by_node_pair: dict[tuple[int, int], list[int]] = {}
    outgoing_links_by_node_pair: dict[tuple[int, int], list[int]] = {}

    # Match incoming turns using the tail and head of each graph link's last
    # network link, and outgoing turns using those of its first network link.
    # These node pairs correspond to (from_node, via_node) and (via_node,
    # to_node) in the turn restrictions.
    for graph_link, (first_link, last_link) in enumerate(zip(first_network_links, last_network_links, strict=True)):
        incoming_node_pair = (network.link_tails[last_link], network.link_heads[last_link])
        outgoing_node_pair = (network.link_tails[first_link], network.link_heads[first_link])

        incoming_links_by_node_pair.setdefault(incoming_node_pair, []).append(graph_link)
        outgoing_links_by_node_pair.setdefault(outgoing_node_pair, []).append(graph_link)

    turn_node_indices = [
        network.node_index(turn_restrictions[column].to_numpy()) for column in ("from_node", "via_node", "to_node")
    ]
    normalised_penalties = [normalise_turn_penalty(penalty) for penalty in turn_restrictions["penalty"]]

    # Return {(incoming_graph_link_index, outgoing_graph_link_index): penalty}.
    turn_penalties: dict[tuple[int, int], float] = {}

    for from_node, via_node, to_node, penalty in zip(*turn_node_indices, normalised_penalties, strict=True):
        for incoming_link in incoming_links_by_node_pair.get((from_node, via_node), ()):
            for outgoing_link in outgoing_links_by_node_pair.get((via_node, to_node), ()):
                _record_turn_penalty(turn_penalties, incoming_link, outgoing_link, penalty)

    return turn_penalties


def _add_centroid_turn_prohibitions(
    turn_penalties: dict[tuple[int, int], float],
    node_link_offsets: NDArray[np.int64],
    link_heads: NDArray[np.intp],
    centroid_count: int,
) -> None:
    """Generate prohibited turns through centroid nodes, preserving connector policy."""
    for incoming_link in np.flatnonzero(link_heads < centroid_count):
        centroid_node = link_heads[incoming_link]
        for outgoing_link in range(node_link_offsets[centroid_node], node_link_offsets[centroid_node + 1]):
            _record_turn_penalty(turn_penalties, incoming_link, outgoing_link, float("inf"))


def _build_turn_penalty_csr(
    link_count: int,
    turn_penalties: dict[tuple[int, int], float],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Build sparse explicit-turn CSR."""
    # Sorted rows support predictable linear scans and early breaks in the kernel.
    sorted_link_pairs = sorted(turn_penalties)
    turn_offsets = np.zeros(link_count + 1, dtype=np.int64)
    for incoming_link, _ in sorted_link_pairs:
        turn_offsets[incoming_link + 1] += 1
    np.cumsum(turn_offsets, out=turn_offsets)

    outgoing_links = np.array([outgoing_link for _, outgoing_link in sorted_link_pairs], dtype=np.int64)
    penalty_values = np.array([turn_penalties[link_pair] for link_pair in sorted_link_pairs], dtype=np.float64)
    return turn_offsets, outgoing_links, penalty_values


def make_routing_context(
    mapping: NetworkGraphMapping,
    costs: NDArray[np.float64],
    *,
    turn_restrictions: pd.DataFrame | None = None,
    centroid_count: int = 0,
    block_centroid_flows: bool = True,
    allow_uturns: bool = False,
) -> GraphContext:
    """
    Build one context from a mapping and costs in directed network order.
    """
    if not isinstance(mapping, NetworkGraphMapping):
        raise ValueError("mapping must be a GraphMapping")
    if isinstance(centroid_count, bool) or not isinstance(centroid_count, int):
        raise ValueError("centroid_count must be an integer")
    if not 0 <= centroid_count <= mapping.node_count:
        raise ValueError("centroid_count must be between zero and the graph node count")
    for option_name, option_value in (
        ("block_centroid_flows", block_centroid_flows),
        ("allow_uturns", allow_uturns),
    ):
        if not isinstance(option_value, bool):
            raise ValueError(f"{option_name} must be a boolean")

    costs = np.asarray(costs)
    if costs.shape != (mapping.network.link_count,) or costs.dtype != np.float64:
        raise ValueError("Costs must be a float64 vector in directed network order")
    if np.isnan(costs).any() or np.any(costs < 0):
        raise ValueError("Graph costs must be nonnegative and cannot contain NaN")
    if turn_restrictions is not None:
        if not isinstance(turn_restrictions, pd.DataFrame):
            raise ValueError("turn_restrictions must be a DataFrame")
        if not {"from_node", "via_node", "to_node", "penalty"}.issubset(turn_restrictions.columns):
            raise ValueError("Turn restrictions require from_node, via_node, to_node and penalty columns")

    first_network_links = mapping.expansion_links[mapping.expansion_offsets[:-1]]
    last_network_links = mapping.expansion_links[mapping.expansion_offsets[1:] - 1]
    network = mapping.network
    graph_link_tails = mapping.network_node_to_graph[network.link_tails[first_network_links]]
    graph_link_heads = mapping.network_node_to_graph[network.link_heads[last_network_links]]
    if np.any(graph_link_tails[1:] < graph_link_tails[:-1]):
        raise ValueError("Graph links must be grouped by their starting node")

    has_turn_restrictions = turn_restrictions is not None and not turn_restrictions.empty
    turn_penalties = (
        _map_explicit_turn_penalties(mapping, turn_restrictions, first_network_links, last_network_links)
        if has_turn_restrictions
        else {}
    )

    # All supplied arguments are valid before creating owned context buffers.
    node_link_offsets = np.empty(mapping.node_count + 1, dtype=np.int64)
    node_link_offsets[0] = 0
    np.cumsum(np.bincount(graph_link_tails, minlength=mapping.node_count), out=node_link_offsets[1:])

    routing_costs = mapping.aggregate_costs(costs)
    routing_costs.flags.writeable = False

    if not has_turn_restrictions:
        return NodeBasedContext(
            node_link_offsets,
            graph_link_heads,
            routing_costs,
            blocked_centroid_count=centroid_count if block_centroid_flows else 0,
            centroid_count=centroid_count,
            block_centroid_flows=block_centroid_flows,
            allow_uturns=allow_uturns,
        )

    # A single bidirectional connector also needs a ban: explicit finite turn
    # penalties can permit U-turns even when path U-turns are otherwise disabled.
    if block_centroid_flows:
        _add_centroid_turn_prohibitions(turn_penalties, node_link_offsets, graph_link_heads, centroid_count)

    turn_offsets, outgoing_links, penalty_values = _build_turn_penalty_csr(mapping.link_count, turn_penalties)

    return TurnBasedContext(
        node_link_offsets,
        graph_link_heads,
        routing_costs,
        turn_fs=turn_offsets,
        turn_to_links=outgoing_links,
        turn_penalties=penalty_values,
        allow_uturns=allow_uturns,
        blocked_centroid_count=0,
        centroid_count=centroid_count,
        block_centroid_flows=block_centroid_flows,
        first_nodes=network.link_heads[first_network_links],
        last_nodes=network.link_tails[last_network_links],
    )
