from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aequilibrae.paths.cython.graph_building import compress_links, find_dead_end_links, order_link_expansions
from aequilibrae.paths.mapping import GraphMapping, NetworkMapping


def normalise_turn_penalty(penalty: object) -> float:
    """Treat missing penalties as banned turns and reject negative costs."""
    if penalty is None or pd.isna(penalty):
        return float("inf")

    try:
        value = float(penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Turn penalties must be None/NaN/+inf for a prohibition or a non-negative finite cost"
        ) from exc

    if value < 0:
        raise ValueError("Negative turn penalties are not allowed. Use None/NaN/+inf for a prohibition")

    return value if np.isfinite(value) else float("inf")


def _directed_arc_positions(
    directions: NDArray[np.int8],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Return physical-link positions and allowed direction slots, in network order.

    Both returned arrays have one entry per available directed arc. For
    example, directions ``[0, 1, -1]`` produce positions ``[0, 0, 1, 2]`` and
    slots ``[0, 1, 0, 1]``. Slot 0 is a-to-b (+1). Slot 1 is b-to-a (-1).
    """
    return np.nonzero(np.column_stack((directions != -1, directions != 1)))


def _physical_links_without_dead_ends(
    physical_a_nodes: NDArray[np.int64],
    physical_b_nodes: NDArray[np.int64],
    physical_directions: NDArray[np.int8],
    node_count: int,
    protected_node_indices: NDArray[np.int64],
    allow_uturns: bool,
) -> NDArray[np.intp]:
    """
    Return the physical-link row positions left after removing directed
    dead ends.
    """
    physical_link_positions, direction_slots = _directed_arc_positions(physical_directions)
    arc_tails = np.where(
        direction_slots == 0,
        physical_a_nodes[physical_link_positions],
        physical_b_nodes[physical_link_positions],
    )
    arc_heads = np.where(
        direction_slots == 0,
        physical_b_nodes[physical_link_positions],
        physical_a_nodes[physical_link_positions],
    )

    # Construct a forward start for the dead end removal
    forward_star_order = np.argsort(arc_tails, kind="stable")
    forward_star = np.empty(node_count + 1, dtype=np.int64)
    forward_star[0] = 0
    np.cumsum(np.bincount(arc_tails, minlength=node_count), out=forward_star[1:])
    burnt_arc_mask = find_dead_end_links(
        forward_star,
        arc_heads[forward_star_order],
        protected_node_indices,
        allow_uturns=allow_uturns,
    )

    # ``burnt_arc_mask`` gives the removed arcs in the sorted order. We need to
    # map these back to physical rows, then return the row positions so we can
    # know which links in the input arrays to keep.  Both directions of a
    # bidirectional link are removed together, so it is safe to remove the
    # whole physical link.
    retained_physical_rows = np.ones(len(physical_directions), dtype=bool)
    burnt_physical_rows = physical_link_positions[forward_star_order][burnt_arc_mask]
    retained_physical_rows[burnt_physical_rows] = False
    return np.flatnonzero(retained_physical_rows)


def _group_directed_arcs_by_graph_link(
    physical_a_nodes: NDArray[np.int64],
    physical_b_nodes: NDArray[np.int64],
    physical_link_positions: NDArray[np.intp],
    direction_slots: NDArray[np.intp],
    physical_to_chain: NDArray[np.int64],
    physical_chain_orientation: NDArray[np.int8],
    chain_start_nodes: NDArray[np.int64],
    chain_end_nodes: NDArray[np.int64],
    chain_allowed_traversal: NDArray[np.int64],
) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.int64], NDArray[np.int64]]:
    """
    Assign each directed source arc to its graph link and return graph endpoints.
    """
    # Encode (topology row, direction slot) as 2 * row + slot. Slot 0 is
    # forward and 1 is backward. Number physical rows first, then chain rows.
    # ``physical_link_positions`` and ``direction_slots`` have one entry per
    # source arc. Each position is a row index and each slot is 0 or 1.
    group_keys = 2 * physical_link_positions + direction_slots
    retained_arc_mask = np.ones(len(physical_link_positions), dtype=bool)
    contracted_arc_mask = physical_to_chain[physical_link_positions] >= 0
    contracted_chain_indices = physical_to_chain[physical_link_positions[contracted_arc_mask]]

    # Work out whether each source arc runs forwards or backwards along its chain.
    source_arc_orientation = np.where(direction_slots[contracted_arc_mask] == 0, 1, -1)
    chain_travel_direction = (
        source_arc_orientation * physical_chain_orientation[physical_link_positions[contracted_arc_mask]]
    )
    group_keys[contracted_arc_mask] = 2 * (len(physical_a_nodes) + contracted_chain_indices) + (
        chain_travel_direction == -1
    )

    # ``physical_to_chain`` and ``physical_chain_orientation`` have one value
    # per physical row. A chain ID of -1 means the link is not part of a chain.
    # For links in a chain, orientation +1 means a-to-b follows the chain
    # forwards.  Orientation -1 means it runs backwards. Ignore orientation for
    # other links.  The other arrays give each chain's start and end nodes and
    # allowed direction: +1 forwards, -1 backwards, 0 both ways, or -999
    # neither way. Keep an arc only if travel is allowed along the whole chain
    # in that direction.
    retained_arc_mask[contracted_arc_mask] = (chain_allowed_traversal[contracted_chain_indices] == 0) | (
        chain_allowed_traversal[contracted_chain_indices] == chain_travel_direction
    )

    # ``unique_group_keys`` has one encoded key per graph
    # link. ``arc_group_indices`` gives each kept arc a group number, starting
    # from zero.  The endpoint arrays follow the sorted key order.
    unique_group_keys, arc_group_indices = np.unique(group_keys[retained_arc_mask], return_inverse=True)
    topology_tails = np.concatenate((physical_a_nodes, chain_start_nodes))
    topology_heads = np.concatenate((physical_b_nodes, chain_end_nodes))
    group_is_backward = unique_group_keys % 2 == 1
    topology_rows = unique_group_keys // 2
    graph_link_tails = np.where(group_is_backward, topology_heads[topology_rows], topology_tails[topology_rows])
    graph_link_heads = np.where(group_is_backward, topology_tails[topology_rows], topology_heads[topology_rows])
    return retained_arc_mask, arc_group_indices, graph_link_tails, graph_link_heads


def build_graph_mapping(
    network: NetworkMapping,
    links: pd.DataFrame,
    *,
    centroids: NDArray | None,
    turn_restrictions: pd.DataFrame | None,
    compact: bool,
    remove_dead_ends: bool,
    allow_uturns: bool,
) -> GraphMapping:
    """
    Build graph-node mappings and ordered source-link expansions.

    The input ``links`` are physical source rows for the selected mode. We
    convert node IDs to node indices before removing dead ends or merging
    chains.

    The returned ``GraphMapping`` to the directed links in the shared
    ``network`` mapping.
    """

    # Look up network indices for the endpoints and protected nodes.
    # Keep external link IDs for later lookup. Each physical_* array has one value
    # per input row, in the same row order.
    physical_a_nodes = network.node_index(links["a_node"].to_numpy()).astype(np.int64, copy=False)
    physical_b_nodes = network.node_index(links["b_node"].to_numpy()).astype(np.int64, copy=False)
    physical_directions = links["direction"].to_numpy(dtype=np.int8)
    physical_link_ids = links["link_id"].to_numpy()
    centroid_node_indices = (
        np.empty(0, dtype=np.int64) if centroids is None else network.node_index(centroids).astype(np.int64, copy=False)
    )
    turn_via_node_indices = (
        np.empty(0, dtype=np.int64)
        if turn_restrictions is None
        else network.node_index(turn_restrictions["via_node"].to_numpy()).astype(np.int64, copy=False)
    )
    protected_node_indices = np.union1d(centroid_node_indices, turn_via_node_indices)

    # The dead-end helper returns the physical row positions to keep.
    # Use these positions to filter all physical-link arrays in the same way.
    if remove_dead_ends:
        retained_physical_rows = _physical_links_without_dead_ends(
            physical_a_nodes,
            physical_b_nodes,
            physical_directions,
            network.node_count,
            protected_node_indices,
            allow_uturns,
        )
        physical_a_nodes = physical_a_nodes[retained_physical_rows]
        physical_b_nodes = physical_b_nodes[retained_physical_rows]
        physical_directions = physical_directions[retained_physical_rows]
        physical_link_ids = physical_link_ids[retained_physical_rows]

    # ``physical_to_chain`` and ``physical_chain_orientation`` have one value
    # per remaining row. A chain ID of -1 means the link is not in a chain.
    # For links in a chain, orientation +1 means a-to-b, -1 means it runs
    # backwards. The other arrays give each chain's start and end node indices
    # and allowed direction: +1 forwards, -1 backwards, 0 both ways, or -999
    # neither way.  Without compaction, there are no chains and all chain IDs
    # are -1.
    if compact:
        (
            physical_to_chain,
            physical_chain_orientation,
            chain_start_nodes,
            chain_end_nodes,
            chain_allowed_traversal,
        ) = compress_links(
            physical_a_nodes,
            physical_b_nodes,
            physical_directions,
            network.node_count,
            protected_node_indices,
            allow_uturns=allow_uturns,
        )
    else:
        physical_to_chain = np.full(len(physical_a_nodes), -1, dtype=np.int64)
        physical_chain_orientation = np.zeros(len(physical_a_nodes), dtype=np.int8)
        chain_start_nodes = np.empty(0, dtype=np.int64)
        chain_end_nodes = np.empty(0, dtype=np.int64)
        chain_allowed_traversal = np.empty(0, dtype=np.int64)

    # Expand each physical row into its available directed arcs, then assign arcs
    # to graph links (single physical links or compressed chains).
    physical_link_positions, direction_slots = _directed_arc_positions(physical_directions)
    (
        retained_arc_mask,
        arc_group_indices,
        graph_link_tails,
        graph_link_heads,
    ) = _group_directed_arcs_by_graph_link(
        physical_a_nodes,
        physical_b_nodes,
        physical_link_positions,
        direction_slots,
        physical_to_chain,
        physical_chain_orientation,
        chain_start_nodes,
        chain_end_nodes,
        chain_allowed_traversal,
    )

    # ``network_link_indices`` contains one directed-network index per retained
    # arc.  Use these to look up link IDs and endpoints in the shared mapping.
    # Costs and other attributes can use the same order, but are stored
    # separately.  These are not physical row positions or external link IDs.
    arc_directions = np.where(direction_slots == 0, 1, -1)
    network_link_indices = network.link_index(physical_link_ids[physical_link_positions], arc_directions)[
        retained_arc_mask
    ]
    source_arc_tails = network.link_tails[network_link_indices].astype(np.int64, copy=False)
    source_arc_heads = network.link_heads[network_link_indices].astype(np.int64, copy=False)

    # Put centroids first in supplied demand order, even if isolated.  Then add
    # the other endpoints of kept graph links, sorted by network index.
    retained_endpoint_nodes = np.union1d(graph_link_tails, graph_link_heads)
    non_centroid_nodes = np.setdiff1d(retained_endpoint_nodes, centroid_node_indices)
    graph_node_to_network = np.concatenate((centroid_node_indices, non_centroid_nodes))
    network_node_to_graph = np.full(network.node_count, -1, dtype=np.intp)
    network_node_to_graph[graph_node_to_network] = np.arange(len(graph_node_to_network), dtype=np.intp)

    # Sort groups by tail then head, using graph node indices.
    forward_star_link_order = np.lexsort(
        (
            np.arange(len(graph_link_tails)),
            network_node_to_graph[graph_link_heads],
            network_node_to_graph[graph_link_tails],
        )
    )
    # Each pair of offsets marks a nonempty source path for one graph link.
    # Returned positions index ``network_link_indices``, not the original
    # DataFrame or external link IDs.
    expansion_offsets, ordered_source_arc_positions = order_link_expansions(
        arc_group_indices,
        source_arc_tails,
        source_arc_heads,
        graph_link_tails,
        forward_star_link_order,
    )
    ordered_network_link_indices = network_link_indices[ordered_source_arc_positions]
    return GraphMapping(
        network,
        graph_node_to_network,
        expansion_offsets,
        ordered_network_link_indices,
    )
