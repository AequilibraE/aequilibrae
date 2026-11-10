"""Standalone network and compression mappings, independent of routing contexts.

Network IDs are external identifiers; every internal index is zero-based. No
array is sized by an external ID. Existing graph consumers do not yet use these
classes.
"""

from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd

from aequilibrae.paths.cython.mapping import aggregate_costs, expand_links, project_loads
from aequilibrae.utils.cython.array_hash_map import ArrayHashMap


def _integers(values, name, *, scalar=True):
    values = np.asarray(values)
    if values.ndim not in ((0, 1) if scalar else (1,)):
        raise ValueError(f"{name} must be {'scalar or ' if scalar else ''}one-dimensional")
    # Empty Python sequences (and empty DataFrame columns) have no dtype to infer.
    if values.ndim == 1 and not values.size:
        return np.empty(0, dtype=np.intp)
    if values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, not booleans or floating-point values")
    return values


def _ids(values, name, *, scalar=True):
    values = _integers(values, name, scalar=scalar)
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    return values.astype(np.uint64, copy=False)


def _indices(values, count, name):
    values = _integers(values, name)
    if np.any(values < 0) or np.any(values >= count):
        raise ValueError(f"{name} must be between 0 and {count - 1}")
    return values.astype(np.intp, copy=False)


def _lookup(mapping, ids):
    if ids.ndim == 0:
        return int(mapping[int(ids)])
    return mapping[np.ascontiguousarray(ids)].astype(np.intp)


def _present(values, message):
    if np.any(values < 0):
        raise KeyError(message)
    return int(values) if np.ndim(values) == 0 else values


def _snapshot(instance, **arrays):
    for name, values in arrays.items():
        values = np.array(values, copy=True, order="C")
        values.flags.writeable = False
        object.__setattr__(instance, name, values)


@dataclass(frozen=True, eq=False)
class NetworkMapping:
    """Snapshot network identity from a link DataFrame, shared between graphs.

    :Arguments:
        **links** (:obj:`pandas.DataFrame`): Columns ``link_id``, ``a_node``,
        ``b_node`` and ``direction``. IDs must be nonnegative integers and link
        IDs must be unique. Direction is 0 for bidirectional, +1 for a-to-b and
        -1 for b-to-a. Additional columns are ignored; no mode filtering occurs.

    Physical links follow supplied row order, irrespective of the DataFrame's
    index labels. Directed links follow that same order, with +1 before -1 where
    both exist. Nodes follow first appearance in the row-wise (a_node, b_node)
    pairs. ``link_ids`` and ``directions`` describe directed link order;
    ``physical_link_ids`` describes input row order.

    Arrays are read-only copies. Later changes to the DataFrame have no effect.
    ID lookups accept integers or one-dimensional NumPy integer arrays and raise
    ``KeyError`` for unknown IDs or unavailable directions.
    """

    links: InitVar[pd.DataFrame]
    physical_link_ids: np.ndarray = field(init=False, repr=False)
    node_ids: np.ndarray = field(init=False, repr=False)
    link_ids: np.ndarray = field(init=False, repr=False)
    directions: np.ndarray = field(init=False, repr=False)
    link_tails: np.ndarray = field(init=False, repr=False)
    link_heads: np.ndarray = field(init=False, repr=False)
    _directed_links: np.ndarray = field(init=False, repr=False)
    _link_indices: ArrayHashMap = field(init=False, repr=False)
    _node_indices: ArrayHashMap = field(init=False, repr=False)

    def __post_init__(self, links):
        required = {"link_id", "a_node", "b_node", "direction"}
        if missing := required - set(links.columns):
            raise ValueError(f"Network is missing required columns: {sorted(missing)}")

        physical_ids = _ids(links.link_id.to_numpy(), "link_id", scalar=False)
        if not pd.Index(physical_ids).is_unique:
            raise ValueError("Network link IDs must be unique")

        a_nodes = _ids(links.a_node.to_numpy(), "a_node", scalar=False)
        b_nodes = _ids(links.b_node.to_numpy(), "b_node", scalar=False)
        direction = _integers(links.direction.to_numpy(), "direction", scalar=False)
        if not np.all(np.isin(direction, [-1, 0, 1])):
            raise ValueError("Network directions must be -1, 0 or +1")

        endpoints = np.column_stack((a_nodes, b_nodes))
        node_ids = pd.unique(endpoints.reshape(-1))
        node_indices = ArrayHashMap(node_ids, np.arange(len(node_ids), dtype=np.uint64))
        link_indices = ArrayHashMap(physical_ids, np.arange(len(physical_ids), dtype=np.uint64))
        physical, slots = np.nonzero(np.column_stack((direction != -1, direction != 1)))
        directed = np.full((len(physical_ids), 2), -1, dtype=np.intp)
        directed[physical, slots] = np.arange(len(physical), dtype=np.intp)
        tails = _lookup(node_indices, endpoints[physical, slots])
        heads = _lookup(node_indices, endpoints[physical, 1 - slots])
        _snapshot(
            self,
            physical_link_ids=physical_ids,
            node_ids=node_ids,
            link_ids=physical_ids[physical],
            directions=np.where(slots == 0, 1, -1).astype(np.int8),
            link_tails=tails,
            link_heads=heads,
            _directed_links=directed,
        )
        object.__setattr__(self, "_node_indices", node_indices)
        object.__setattr__(self, "_link_indices", link_indices)

    @property
    def link_count(self):
        """Number of directed network links."""
        return len(self.link_ids)

    @property
    def node_count(self):
        return len(self.node_ids)

    def node_index(self, node_ids):
        """Map external node IDs to network indices, preserving query order."""
        return _lookup(self._node_indices, _ids(node_ids, "node_ids"))

    def link_index(self, link_ids, directions):
        """Map IDs and explicit +1/-1 directions to directed network indices.

        Scalars and one-dimensional arrays broadcast together. A scalar pair
        returns a Python integer; array queries return an array of indices.
        Direction 0 is not a lookup wildcard.
        """
        link_ids = _ids(link_ids, "link_ids")
        directions = _integers(directions, "directions")
        if not np.all(np.isin(directions, [-1, 1])):
            raise ValueError("Lookup directions must be -1 or +1")

        link_ids, directions = np.broadcast_arrays(link_ids, directions)
        physical = _lookup(self._link_indices, link_ids)
        indices = self._directed_links[physical, (directions == -1).astype(np.intp)]
        return _present(indices, "Network link direction is unavailable")


@dataclass(frozen=True, eq=False)
class GraphMapping:
    """Keep a graph's network correspondence and ordered expansions together.

    :Arguments:
        **network** (:obj:`NetworkMapping`): Shared whole-network snapshot.

        **graph_node_to_network** (:obj:`np.ndarray`): Retained network node
        indices in graph node order, including any isolated retained nodes.

        **expansion_offsets**, **expansion_links** (:obj:`np.ndarray`): Ordered
        directed network link indices for each graph link. Graph link ``i``
        expands to ``expansion_links[expansion_offsets[i]:expansion_offsets[i+1]]``.
        Offsets start at zero, end at the expansion length and have one more
        entry than the graph link count. Every graph link has a nonempty,
        connected expansion whose endpoints are retained nodes. Each directed
        network link may belong to at most one graph link.

    Construction copies the arrays and derives inverse mappings. Unmapped
    entries in ``network_link_to_graph`` and ``network_node_to_graph`` are -1;
    Python lookups raise ``KeyError`` instead of returning that sentinel.
    Unknown, removed and mode-excluded elements are not distinguished.

    All mapping arrays are contiguous and read-only. Internal Cython callers
    can bind the dense-index arrays to ``const Py_ssize_t[::1]`` views under the
    GIL, then use them without the GIL. External-ID hashing stays at the boundary.
    """

    network: NetworkMapping
    graph_node_to_network: np.ndarray = field(repr=False)
    expansion_offsets: np.ndarray = field(repr=False)
    expansion_links: np.ndarray = field(repr=False)
    network_link_to_graph: np.ndarray = field(init=False, repr=False)
    network_node_to_graph: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        nodes = _indices(self.graph_node_to_network, self.network.node_count, "graph_node_to_network")
        links = _indices(self.expansion_links, self.network.link_count, "expansion_links")
        offsets = _indices(self.expansion_offsets, links.size + 1, "expansion_offsets")

        if nodes.ndim != 1 or links.ndim != 1 or offsets.ndim != 1:
            raise ValueError("Mapping arrays must be one-dimensional")
        if not len(offsets) or offsets[0] != 0 or offsets[-1] != len(links) or np.any(np.diff(offsets) <= 0):
            raise ValueError("Expansion offsets must span the links with a nonempty segment per graph link")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Retained network nodes must be unique")
        if len(np.unique(links)) != len(links):
            raise ValueError("Each directed network link may belong to at most one graph link")

        node_inverse = np.full(self.network.node_count, -1, dtype=np.intp)
        node_inverse[nodes] = np.arange(len(nodes), dtype=np.intp)
        link_inverse = np.full(self.network.link_count, -1, dtype=np.intp)
        link_inverse[links] = np.repeat(np.arange(len(offsets) - 1, dtype=np.intp), np.diff(offsets))
        self._validate_chains(links, offsets, node_inverse)
        _snapshot(
            self,
            graph_node_to_network=nodes,
            expansion_offsets=offsets,
            expansion_links=links,
            network_node_to_graph=node_inverse,
            network_link_to_graph=link_inverse,
        )

    def _validate_chains(self, links, offsets, node_inverse):
        if not len(links):
            return
        tails, heads = self.network.link_tails[links], self.network.link_heads[links]
        if np.any(node_inverse[tails[offsets[:-1]]] < 0) or np.any(node_inverse[heads[offsets[1:] - 1]] < 0):
            raise ValueError("Expansion endpoints must be retained graph nodes")
        disconnected = heads[:-1] != tails[1:]
        disconnected[offsets[1:-1] - 1] = False
        if np.any(disconnected):
            raise ValueError("Expansion links must form a connected chain in traversal order")

    @property
    def link_count(self):
        return len(self.expansion_offsets) - 1

    @property
    def node_count(self):
        return len(self.graph_node_to_network)

    def graph_link(self, link_ids, directions):
        """Find graph link indices from network IDs and explicit +1/-1 directions.

        Scalar and array inputs broadcast as in :meth:`NetworkMapping.link_index`.
        Any missing correspondence raises ``KeyError`` for the whole query.
        """
        indices = self.network.link_index(link_ids, directions)
        return _present(self.network_link_to_graph[indices], "Network link has no graph mapping")

    def graph_node(self, node_ids):
        """Find graph node indices from network IDs; missing mappings raise ``KeyError``."""
        indices = self.network.node_index(node_ids)
        return _present(self.network_node_to_graph[indices], "Network node has no graph mapping")

    def network_node(self, graph_nodes):
        """Return external network node IDs for scalar or array graph indices."""
        indices = _indices(graph_nodes, self.node_count, "graph_nodes")
        ids = self.network.node_ids[self.graph_node_to_network[indices]]
        return int(ids) if ids.ndim == 0 else ids

    def expand_links(self, graph_links):
        """Return flat ``(link_ids, directions)`` arrays in traversal order.

        Accept a scalar or one-dimensional array of graph link indices. Multiple
        expansions are concatenated in query order, preserving duplicates;
        queries need not themselves describe a connected path. Even scalar
        queries return arrays. Empty queries return two empty arrays.
        """
        indices = _indices(graph_links, self.link_count, "graph_links")
        links = expand_links(np.ascontiguousarray(np.atleast_1d(indices)), self.expansion_offsets, self.expansion_links)
        return self.network.link_ids[links], self.network.directions[links]

    def aggregate_costs(self, network_costs, *, out=None):
        """
        Sum directed network costs along each graph link's ordered expansion.

        :Arguments:
            **network_costs** (:obj:`np.ndarray`): ``float32`` or ``float64``
                values shaped ``(network_links,)`` or ``(network_links,
                fields)``, in this mapping's directed network order.  Read-only
                and non-contiguous arrays are accepted.

            **out** (:obj:`np.ndarray`, optional): Writable output shaped
                ``(graph_links,)`` or ``(graph_links, fields)``, matching the
                input's dimensionality and dtype. It must not overlap the
                input. Strided arrays are accepted. Existing values are
                overwritten, not accumulated.

        :Returns:
            A new array, or **out** itself when supplied. Both accumulation and
            output use the input dtype. Each field is summed independently in
            traversal order; unmapped network links are ignored. Costs are not
            otherwise validated: retained infinities and NaNs participate in
            ordinary floating-point addition. This also suits additive skim
            fields.
        """
        costs = np.asarray(network_costs)
        if costs.ndim not in (1, 2) or costs.shape[0] != self.network.link_count:
            raise ValueError("Costs must have one row per directed network link and one or two dimensions")
        if costs.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Costs must have native-endian float32 or float64 dtype")

        shape = (self.link_count, *costs.shape[1:])

        if out is None:
            out = np.empty(shape, dtype=costs.dtype)
        else:
            if not isinstance(out, np.ndarray) or out.shape != shape:
                raise ValueError(f"out must be a NumPy array with shape {shape}")
            if out.dtype != costs.dtype:
                raise ValueError("out dtype must match the input cost dtype")
            if not out.flags.writeable:
                raise ValueError("out must be writable")
            if np.shares_memory(out, costs):
                raise ValueError("out must not overlap the input costs")

        source = costs[:, None] if costs.ndim == 1 else costs
        target = out[:, None] if out.ndim == 1 else out
        aggregate_costs(self.expansion_offsets, self.expansion_links, source, target)
        return out

    def project_loads(self, graph_loads):
        """Copy graph loads to directed network order, filling unmapped rows with zero.

        Accept real integer or floating-point arrays shaped ``(graph_links,)``
        or ``(graph_links, classes)``. Return a new array preserving dtype and
        trailing dimensions. A graph link's load is copied to every constituent
        network link, not divided between them. Non-contiguous inputs are accepted.
        """
        loads = np.asarray(graph_loads)
        if loads.ndim not in (1, 2) or loads.shape[0] != self.link_count:
            raise ValueError("Loads must have one row per graph link and one or two dimensions")
        if loads.dtype.kind not in "iuf":
            raise ValueError("Loads must contain real integer or floating-point values")

        return project_loads(self.network_link_to_graph, np.ascontiguousarray(loads))
