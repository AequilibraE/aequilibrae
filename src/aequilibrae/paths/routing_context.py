from typing import TYPE_CHECKING

import numpy as np

from aequilibrae.paths.cython.context import GraphContext

if TYPE_CHECKING:
    from aequilibrae.paths.graph import Graph


def _copy_ids(values: np.ndarray, count: int, name: str) -> np.ndarray:
    """Copy node IDs to a C-contiguous, read-only array."""
    if values is not None:
        array = np.asarray(values, dtype=np.int64, order="C")
        if not array.flags.c_contiguous:
            array = np.array(array, dtype=np.int64, order="C", copy=True)
        array.flags.writeable = False
        return array


class GraphMapping:
    """
    Map node IDs to routing-context positions and retrieve path nodes in network order.

    Node IDs in a graph are external identifiers: they may be sparse, negative, or
    arbitrary.  The context node map shrinks these to a dense 0..N index suitable for
    array subscripting and the forward-star representation.

    Edges are given to the context in order of tail node, then head node; ``link_ids``
    and ``directions`` in the same order describe what each edge represents.
    """

    def __init__(self, context: GraphContext, node_ids: np.ndarray, link_ids: np.ndarray, directions: np.ndarray):
        self.context = context
        self.node_ids = _copy_ids(node_ids, None, "node_ids")
        self.link_ids = _copy_ids(link_ids, None, "link_ids")
        self.directions = _copy_ids(directions, None, "directions")

    @property
    def centroids(self):
        return self.node_ids[self.context.centroids]

    def path_nodes(self, origin, links):
        """Map compact link indices and origin ID to network nodes.

        Paths are stored as link sequences. To reconstruct the nodes from a link
        sequence, map links.a_node and link.b_node from the edge table and then
        translate both back to external node IDs.

        Link indices in ``links`` are subscripts into the edge array given to
        ``prepare_routing()``, which are the same subscripts used by search results.
        """
        if links.size == 0:
            return np.array([origin], dtype=np.int64)

        heads = self.context.heads
        nodes = np.empty(len(links) + 1, dtype=np.int64)
        nodes[0] = self.node_ids[origin]
        nodes[1:] = self.node_ids[heads[links]]

        return nodes


def make_routing_context(
    graph: "Graph", costs: np.ndarray | None = None, *, compact: bool = False
) -> GraphContext:
    """Copy topology from Graph into a full or compact routing context.

    Without costs, copy the graph's current cost buffer too. Supplied costs are
    borrowed so assignment can update its own buffer between searches. No Graph
    arrays are changed or retained by the context when costs are omitted.
    """
    if costs is None:
        costs = np.array(graph.routing.costs, copy=True)
        costs.flags.writeable = False
    return graph.routing.with_costs(costs)
