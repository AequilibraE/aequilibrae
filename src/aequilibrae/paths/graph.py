from __future__ import annotations

import dataclasses
import logging
import pickle
import uuid
import warnings
from abc import ABC
from copy import deepcopy
from datetime import datetime
from os.path import join
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from aequilibrae.paths.connectivity_analysis import disconnected_analysis
from aequilibrae.paths.cython.graph_building import build_compressed_graph, create_compressed_link_network_mapping
from aequilibrae.paths.cython.public_transport import HyperpathGenerating

if TYPE_CHECKING:
    from aequilibrae.paths import PathResults

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NetworkGraphIndices:
    network_ab_idx: np.ndarray
    network_ba_idx: np.ndarray
    graph_ab_idx: np.ndarray
    graph_ba_idx: np.ndarray


def _get_graph_to_network_mapping(lids, direcs):
    num_uncompressed_links = int(np.unique(lids).shape[0])
    indexing = np.zeros(int(lids.max()) + 1, np.uint64)
    indexing[np.unique(lids)[:]] = np.arange(num_uncompressed_links)

    graph_ab_idx = direcs > 0
    graph_ba_idx = direcs < 0
    network_ab_idx = indexing[lids[graph_ab_idx]]
    network_ba_idx = indexing[lids[graph_ba_idx]]
    return NetworkGraphIndices(network_ab_idx, network_ba_idx, graph_ab_idx, graph_ba_idx)


class GraphBase(ABC):  # noqa: B024
    """
    Graph class.

    AequilibraE graphs implement two forms of compression.
        - link contraction, and
        - dead end removal.

    Link contraction creates a topological equivalent graph by contracting sequences of links between nodes
    with degrees of two. This compresses long streams of links, such as along highways or curved roads, into
    single links.

    Dead end removal attempts to remove dead ends and fish spines from the network. It does this based on the
    observation that in a graph with non-negative weights a dead end will only ever appear in the results of a
    short(est) path if the origin or destination is present within that dead end.

    Dead end removal is applied before link contraction and does not create a strictly topological equivalent graph,
    however, all centroids are preserved.

    The compressed graph is used internally.
    """

    def __init__(self, logger=None):
        self.__int_type = np.int64
        self.__float_type = np.float64

        self.required_fields = ["link_id", "a_node", "b_node", "direction", "id"]
        self.__required_types = [self.__int_type, self.__int_type, self.__int_type, np.int8, self.__int_type]
        self.other_fields = ""
        self.mode = ""
        self.date = str(datetime.now())

        self.turn_penalty_dimension = "time"
        self._turn_penalties_master = None
        self._compact_turn_penalties_master = None

        self.description = "No description added so far"
        self.__graph_groupby = None

        self.num_links = -1
        self.num_nodes = -1
        self.num_zones = -1

        self.compact_num_links = -1
        self.compact_num_nodes = -1

        self.network = pd.DataFrame([])  # This method will hold ALL information on the network
        self.graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        self.compact_graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        # These are the fields actually used in computing paths
        self.all_nodes = np.array(0)  # Holds an array with all nodes in the original network
        self.nodes_to_indices = np.array(0, np.int64)  # Holds the reverse of the all_nodes
        self.fs = np.array([])  # This method will hold the forward star for the graph
        self.cost = np.array([])  # This array holds the values being used in the shortest path routine
        self.skims = None

        self.lonlat_index = pd.DataFrame([])  # Holds a node_id to lon/lat coord index for nodes within this graph

        self.compact_all_nodes = np.array(0)  # Holds an array with all nodes in the original network
        self.compact_nodes_to_indices = np.array(0)  # Holds the reverse of the all_nodes
        self.compact_fs = np.array([])  # This method will hold the forward star for the graph
        self.compact_cost = np.array([])  # This array holds the values being used in the shortest path routine
        self.compact_skims = None

        self.capacity = np.array([])  # Array holds the capacity for links
        self.free_flow_time = np.array([])  # Array holds the free flow travel time by link

        # sake of the Cython code
        self.skim_fields = []  # List of skim fields to be used in computation
        self.cost_field = False  # Name of the cost field

        self.block_centroid_flows = True
        self.penalty_through_centroids = np.inf

        self.centroids = None  # NumPy array of centroid IDs

        self.g_link_crosswalk = np.array([])  # 4 a link ID in the BIG graph, a corresponding link in the compressed 1

        self.dead_end_links = np.array([], dtype=np.int64)

        self.compressed_link_network_mapping_idx = None
        self.compressed_link_network_mapping_data = None
        self.network_compressed_node_mapping = None

        # Turn restrictions data structures
        self._effective_vias_cache = None  # (key, vias) memo for _compute_effective_turn_vias
        self._turn_restrictions = None  # DataFrame of turn restrictions
        self._allow_uturns_everywhere = False  # Preserve U-turn nodes during graph compression
        self._allow_path_uturns = False  # Allow U-turns during arc-based pathfinding
        self._has_turn_restrictions = False  # Flag to indicate if turn restrictions are active

        # Which skim fields receive the accumulated turn penalty.  Default (empty list) falls
        # back to [cost_field] at runtime, preserving backward-compatible behaviour.
        self.turn_skim_fields: List[str] = []

        # Turn restriction CSR structures for arc-based pathfinding (full graph)
        self.turn_fs = np.array([])
        self.turn_to_arcs = np.array([])
        self.turn_penalties = np.array([])

        # Turn restriction CSR structures for arc-based pathfinding (compact graph)
        self.compact_turn_fs = np.array([])  # Forward star for turn transitions (indexed by arc)
        self.compact_turn_to_arcs = np.array([])  # Target arcs for each turn
        self.compact_turn_penalties = np.array([])  # Penalties for each turn (INFINITY = prohibited)

        # Supernet-id -> compressed-id map, used to aggregate link costs onto the compact
        # graph. Built with the compressed graph and rebuilt if the graph is replaced.
        self._crosswalk = None

        # Hybrid node/arc-state Dijkstra structures
        self.stateful = np.empty(0, dtype=np.uint8)
        self.rep_arc = np.empty(0, dtype=np.int64)
        self.compact_stateful = np.empty(0, dtype=np.uint8)
        self.compact_rep_arc = np.empty(0, dtype=np.int64)

        self._remove_dead_ends = True
        self._turn_topology_signature = None
        self._compact_first_node = np.empty(0, dtype=np.int64)
        self._compact_last_node = np.empty(0, dtype=np.int64)

        self._graph_generation: int = 0
        self._turn_restrictions_generation: int = 0

        # Randomly generate a unique Graph ID randomly
        self._id = uuid.uuid4().hex

    def default_types(self, tp: str):
        """
        Returns the default integer and float types used for computation

        :Arguments:
            **tp** (:obj:`str`): data type. 'int' or 'float'
        """
        if tp == "int":
            return self.__int_type
        elif tp == "float":
            return self.__float_type
        else:
            raise ValueError("It must be either a int or a float")

    def reverse(self):
        g = deepcopy(self)
        g.network = g.network.rename(columns={"a_node": "b_node", "b_node": "a_node"})
        if self._turn_restrictions is not None and len(self._turn_restrictions) > 0:
            rev_tr = self._turn_restrictions.copy()
            rev_tr = rev_tr.rename(columns={"from_node": "to_node", "to_node": "from_node"})
            rev_tr = rev_tr[["from_node", "via_node", "to_node", "penalty"]]
            g._turn_restrictions = rev_tr
        else:
            g._turn_restrictions = None

        g._reprepare(self.centroids)
        g._id = uuid.uuid4().hex
        return g

    def prepare_graph(
        self,
        centroids: Optional[np.ndarray] = None,
        remove_dead_ends: bool = True,
        allow_uturns_everywhere: bool = False,
    ) -> None:
        """
        Prepares the graph for a computation for a certain set of centroids.

        Under the hood, if sets all centroids to have IDs from 1 through **n**,
        which should correspond to the index of the matrix being assigned.

        This is what enables having any node IDs as centroids, and it relies on
        the inference that all links connected to these nodes are centroid
        connectors.

        :Arguments:
            **centroids** (``np.ndarray`` or ``None``, optional): Array with centroid IDs. Mandatory type
                ``Int64``, unique and positive.

            **remove_dead_ends** (``bool``, optional): Whether or not to remove dead ends from the graph.
                Defaults to ``True``.

            **allow_uturns_everywhere** (:obj:`bool`, optional): Allow U-turns at all possible places within the
                graph. This will allow U-turns anywhere that has both an incoming and outgoing link, severely hindering
                the effectiveness of graph compression. This is option should rarely be used. **The U-turns allowed by
                this option will never be taken unless turn penalties are set**. Refer to
                ``graph.set_turn_restrictions`` to allow U-turns at intersections. This options takes effect after dead
                end link removal.
        """
        self._remove_dead_ends = remove_dead_ends
        self._allow_uturns_everywhere = allow_uturns_everywhere
        self._effective_vias_cache = None
        self._graph_generation += 1

        self.__network_error_checking__()

        # Creates the centroids
        if centroids is not None:
            if not np.issubdtype(centroids.dtype, np.integer):
                raise ValueError("Centroids need to be a NumPy array of integers 64 bits")
            if centroids.shape[0] == 0:
                raise ValueError("You need at least one centroid")
            if centroids.min() <= 0:
                raise ValueError("Centroid IDs need to be positive")
            if centroids.shape[0] != np.unique(centroids).shape[0]:
                raise ValueError("Centroid IDs are not unique")
            self.centroids = np.array(centroids, np.uint32)
        else:
            self.centroids = np.array([], np.uint32)

        self.network = self.network.astype(
            {
                "direction": np.int8,
                "a_node": self.__int_type,
                "b_node": self.__int_type,
                "link_id": self.__int_type,
            }
        )

        if self.network.empty:
            self._initialize_empty_topology()
            return

        properties = self._build_directed_graph(self.network, self.centroids)
        self.all_nodes, self.num_nodes, self.nodes_to_indices, self.fs, self.graph = properties

        # We generate IDs that we KNOW will be constant across modes
        if "__supernet_id__" not in self.graph.columns:
            self.graph.sort_values(by=["link_id", "direction"], inplace=True)
            self.graph["__supernet_id__"] = np.arange(self.graph.shape[0]).astype(self.__int_type)
        self.graph.sort_values(by=["a_node", "b_node"], inplace=True)

        self.num_links = self.graph.shape[0]
        self.__build_derived_properties()

        if self.centroids.shape[0]:
            self.__build_compressed_graph(remove_dead_ends)
            self.compact_num_links = self.compact_graph.shape[0]
        else:
            self.__graph_groupby = None
            self.compact_graph = pd.DataFrame([])
            self.compact_num_links = 0
            self.compact_all_nodes = np.empty(0, dtype=self.__int_type)
            self.compact_nodes_to_indices = np.empty(0, dtype=np.int64)
            self.compact_fs = np.zeros(1, dtype=self.__int_type)
            self.compact_cost = np.zeros(1, dtype=self.__float_type)
            self.compact_skims = None
            self._compact_first_node = np.empty(0, dtype=np.int64)
            self._compact_last_node = np.empty(0, dtype=np.int64)

        # The cache property should be recalculated when the graph has been re-prepared
        self.compressed_link_network_mapping_idx = None
        self.compressed_link_network_mapping_data = None
        self.network_compressed_node_mapping = None

        self._turn_topology_signature = self._compute_turn_topology_signature()
        self._build_turn_csr_structures()
        self._graph_generation += 1
        self._id = uuid.uuid4().hex

    def _initialize_empty_topology(self) -> None:
        """
        Canonical empty-topology initializer.
        Clears every full, compact, turn, DAG, cost, skim, and winner array
        into a consistent zero-arc state.
        """
        empty_props = self._build_directed_graph(
            self.network, self.centroids if self.centroids is not None else np.empty(0, dtype=self.__int_type)
        )
        self.all_nodes = empty_props[0]
        self.num_nodes = empty_props[1]
        self.nodes_to_indices = empty_props[2]
        self.fs = empty_props[3]
        self.graph = empty_props[4]
        self.graph["__compressed_id__"] = np.empty(0, dtype=np.int64)
        self.graph["__supernet_id__"] = np.empty(0, dtype=self.__int_type)
        self.num_links = 0
        self.cost = np.zeros(0, dtype=self.__float_type)
        if self.skim_fields:
            self.skims = np.zeros((1, len(self.skim_fields) + 1), dtype=self.__float_type)
        else:
            self.skims = np.zeros((1, 1), dtype=self.__float_type)

        if self.centroids is not None and len(self.centroids) > 0:
            self.compact_all_nodes = np.array(self.centroids, copy=True).astype(self.__int_type)
            self.compact_num_nodes = len(self.compact_all_nodes)
            self.compact_nodes_to_indices = np.full(int(self.compact_all_nodes.max()) + 1, -1, dtype=np.int64)
            self.compact_nodes_to_indices[self.compact_all_nodes] = np.arange(self.compact_num_nodes)
        else:
            self.compact_all_nodes = np.empty(0, dtype=self.__int_type)
            self.compact_num_nodes = 0
            self.compact_nodes_to_indices = np.empty(0, dtype=np.int64)

        self.compact_fs = np.zeros(self.compact_num_nodes + 1, dtype=self.__int_type)
        self.compact_graph = pd.DataFrame(columns=["id", "link_id", "a_node", "b_node", "direction"])
        self.compact_num_links = 0
        self.compact_cost = np.zeros(1, dtype=self.__float_type)
        if self.skim_fields and self.compact_num_nodes > 0:
            self.compact_skims = np.zeros((1, len(self.skim_fields) + 1), dtype=self.__float_type)
        else:
            self.compact_skims = None
        self.dead_end_links = np.empty(0, dtype=np.int64)
        self._crosswalk = None
        self.__graph_groupby = None

        # Turn restrictions
        self.turn_fs = np.zeros(1, dtype=self.__int_type)
        self.turn_to_arcs = np.empty(0, dtype=self.__int_type)
        self.turn_penalties = np.empty(0, dtype=self.__float_type)
        self._turn_penalties_master = np.empty(0, dtype=self.__float_type)
        self.compact_turn_fs = np.zeros(1, dtype=self.__int_type)
        self.compact_turn_to_arcs = np.empty(0, dtype=self.__int_type)
        self.compact_turn_penalties = np.empty(0, dtype=self.__float_type)
        self._compact_turn_penalties_master = np.empty(0, dtype=self.__float_type)
        self.stateful = np.empty(0, dtype=np.uint8)
        self.rep_arc = np.empty(0, dtype=np.int64)
        self.compact_stateful = np.empty(0, dtype=np.uint8)
        self.compact_rep_arc = np.empty(0, dtype=np.int64)
        self._has_turn_restrictions = False
        self._compact_first_node = np.empty(0, dtype=np.int64)
        self._compact_last_node = np.empty(0, dtype=np.int64)
        self._turn_topology_signature = ((), bool(self._allow_path_uturns), bool(self._allow_uturns_everywhere))

        self.compressed_link_network_mapping_idx = None
        self.compressed_link_network_mapping_data = None
        self.network_compressed_node_mapping = None
        self._effective_vias_cache = None
        self._graph_generation += 1
        self.__build_derived_properties()
        self._id = uuid.uuid4().hex

    def __build_compressed_graph(self, remove_dead_ends):
        build_compressed_graph(self, remove_dead_ends)
        self._build_crosswalk()

        # We build a groupby to save time later
        self.__graph_groupby = self.graph.groupby(["__compressed_id__"])

    def _restore_computation_state(self) -> None:
        """Reinstalls the cost field, skims and centroid blocking after a re-preparation."""
        if self.cost_field:
            self.set_graph(self.cost_field)
        if self.skim_fields:
            self.set_skimming(self.skim_fields)
        if self.turn_skim_fields:
            self.turn_skim_fields = list(self.turn_skim_fields)
        self.set_blocked_centroid_flows(self.block_centroid_flows)

    def _reprepare(self, centroids) -> None:
        """Rebuilds the graph with its current settings and restores computation state."""
        self.prepare_graph(
            centroids,
            remove_dead_ends=self._remove_dead_ends,
            allow_uturns_everywhere=self._allow_uturns_everywhere,
        )
        self._restore_computation_state()

    def _build_directed_graph(self, network: pd.DataFrame, centroids: np.ndarray):
        all_titles = list(network.columns)

        not_pos = network.loc[network.direction != 1, :]
        not_negs = network.loc[network.direction != -1, :]

        names, types = self.__build_column_names(all_titles)
        neg_names = []
        for name in names:
            if name in not_pos.columns:
                neg_names.append(name)
            elif name + "_ba" in not_pos.columns:
                neg_names.append(name + "_ba")
        not_pos = pd.DataFrame(not_pos, copy=True)[neg_names]
        not_pos.columns = names

        # Swap the a and b nodes of these edges. Direction is used for mapping the graph.graph back
        # to the network. It does not indicate the direction of the link.
        not_pos.loc[:, "direction"] = -1
        aux = np.array(not_pos.a_node.values, copy=True)
        not_pos.loc[:, "a_node"] = not_pos.loc[:, "b_node"]
        not_pos.loc[:, "b_node"] = aux[:]
        del aux

        pos_names = []
        for name in names:
            if name in not_negs.columns:
                pos_names.append(name)
            elif name + "_ab" in not_negs.columns:
                pos_names.append(name + "_ab")
        not_negs = pd.DataFrame(not_negs, copy=True)[pos_names]
        not_negs.columns = names
        not_negs.loc[:, "direction"] = 1

        df = pd.concat([not_negs, not_pos])

        # Now we take care of centroids
        nodes = np.unique(np.hstack((df.a_node.values, df.b_node.values))).astype(self.__int_type)
        present_centroids = np.isin(centroids, nodes, assume_unique=True)
        if not present_centroids.all():
            warnings.warn(
                "Found centroids not present in the graph!\n" + str(centroids[~present_centroids]), stacklevel=2
            )
        nodes = np.setdiff1d(nodes, centroids, assume_unique=True)
        all_nodes = np.hstack((centroids, nodes)).astype(self.__int_type)

        num_nodes = all_nodes.shape[0]
        if num_nodes == 0:
            nodes_to_indices = np.empty(0, dtype=np.int64)
            fs = np.zeros(1, dtype=self.__int_type)
        else:
            nodes_to_indices = np.full(int(all_nodes.max()) + 1, -1, dtype=np.int64)
            nlist = np.arange(num_nodes)
            nodes_to_indices[all_nodes] = nlist

            df.a_node = nodes_to_indices[df.a_node.values]
            df.b_node = nodes_to_indices[df.b_node.values]
            df = df.sort_values(by=["a_node", "b_node"])
            df.index = np.arange(df.shape[0])
            df["id"] = np.arange(df.shape[0])
            fs = np.empty(num_nodes + 1, dtype=self.__int_type)
            fs.fill(-1)
            y, x, _ = np.intersect1d(df.a_node.values, nlist, assume_unique=False, return_indices=True)
            fs[y] = x[:]
            fs[-1] = df.shape[0]
            for i in range(num_nodes, 0, -1):
                if fs[i - 1] == -1:
                    fs[i - 1] = fs[i]

        nans = ", ".join([i for i in df.columns if df[i].isnull().any().any()])
        if nans:
            logger.warning(f"Field(s) {nans} has(ve) at least one NaN value. Check your computations")

        df["link_id"] = df["link_id"].astype(self.__int_type)
        df["b_node"] = df.b_node.values.astype(self.__int_type)
        df["id"] = df.id.values.astype(self.__int_type)
        df["direction"] = df.direction.values.astype(np.int8)

        return all_nodes, num_nodes, nodes_to_indices, fs, df

    def compute_path(
        self,
        origin: int,
        destination: int,
        early_exit: bool = False,
        a_star: bool = False,
        heuristic: Union[str, None] = None,
        *,
        coordinates: pd.DataFrame | None = None,
        heuristic_scale: float | None = None,
    ) -> PathResults:
        """
        Returns the results from path computation result holder.

        :Arguments:
            **origin** (:obj:`int`): origin for the path

            **destination** (:obj:`int`): destination for the path

            **early_exit** (:obj:`bool`): stop constructing the shortest path tree once the destination is found.
            Doing so may cause subsequent calls to 'update_trace' to recompute the tree. Default is ``False``.

            **a_star** (:obj:`bool`): whether or not to use A* over Dijkstra's algorithm.
            When ``True``, 'early_exit' is always ``True``. Default is ``False``.

            **heuristic** (:obj:`str`): ``euclidean`` (default) or ``haversine`` if A* is enabled.

            **coordinates** (:obj:`pandas.DataFrame`, optional): Planar ``x`` and ``y`` columns indexed by
            external node ID, required for Euclidean A*. Haversine uses the graph's longitude/latitude.

            **heuristic_scale** (:obj:`float`): Finite, nonnegative coefficient required for A*.  Use
            ``aequilibrae.paths.estimate_heuristic_scale`` for a conservative bound. A larger scale can give
            non-shortest paths.
        """
        from aequilibrae.paths import PathResults

        res = PathResults(
            self,
            origin,
            destination,
            early_exit=early_exit,
            a_star=a_star,
            heuristic=heuristic,
            coordinates=coordinates,
            heuristic_scale=heuristic_scale,
        )

        return res

    def compute_skims(self, cores: Union[int, None] = None):
        """
        Returns the results from network skimming result holder.

        :Arguments:
            **cores** (:obj:`Union[int, None]`): number of cores (threads) to be used in computation
        """
        from aequilibrae.paths import NetworkSkimming

        skimmer = NetworkSkimming(self)

        if cores is not None:
            skimmer.set_cores(cores)
        skimmer.execute()

        return skimmer

    def exclude_links(self, links: list) -> None:
        """
        Excludes a list of links from a graph by removing them from the network.

        :Arguments:
            **links** (:obj:`list`): List of link IDs to be excluded from the graph
        """
        links_set = set(links)
        filter_mask = self.network.link_id.isin(links_set)
        if filter_mask.sum() != len(links_set):
            logger.warning("At least one link does not exist in the network and therefore cannot be excluded")

        self.network = self.network.loc[~filter_mask, :].copy()

        if self.network.empty:
            self._initialize_empty_topology()
            return

        if self.centroids is not None and self.centroids.shape[0] > 0:
            self._reprepare(self.centroids)
        else:
            if not self.network.empty and self.num_nodes >= 0:
                self._reprepare(None)
            else:
                self._id = uuid.uuid4().hex

    def disconnected_nodes(self) -> np.ndarray:
        """
        Executes strongly connected components analysis on the directed graph

        :Returns:
            **array** (:obj:`np.ndarray`): All nodes disconnected from the main portion of the network
        """
        return disconnected_analysis(self)

    def __build_column_names(self, all_titles: List[str]) -> Tuple[list, list]:
        fields = list(self.required_fields)
        types = list(self.__required_types)
        for column in all_titles:
            if column not in self.required_fields and column[0:-3] not in self.required_fields:
                if column[-3:] == "_ab":
                    if column[:-3] + "_ba" in all_titles:
                        fields.append(column[:-3])
                        types.append(self.network[column].dtype)
                    else:
                        raise ValueError("Field {} exists for ab direction but does not exist for ba".format(column))
                elif column[-3:] == "_ba":
                    if column[:-3] + "_ab" not in all_titles:
                        raise ValueError("Field {} exists for ba direction but does not exist for ab".format(column))
                else:
                    fields.append(column)
                    types.append(self.network[column].dtype)
        return fields, types

    def __build_dtype(self, all_titles) -> list:
        dtype = [
            ("link_id", self.__int_type),
            ("a_node", self.__int_type),
            ("b_node", self.__int_type),
            ("direction", np.int8),
            ("id", self.__int_type),
        ]
        for i in all_titles:
            if i not in self.required_fields and i[0:-3] not in self.required_fields:
                if i[-3:] == "_ab":
                    if i[:-3] + "_ba" in all_titles:
                        dtype.append((i[:-3], self.network[i].dtype))
                    else:
                        raise ValueError("Field {} exists for ab direction but does not exist for ba".format(i))
                elif i[-3:] == "_ba":
                    if i[:-3] + "_ab" not in all_titles:
                        raise ValueError("Field {} exists for ba direction but does not exist for ab".format(i))
                else:
                    dtype.append((i, self.network[i].dtype))
        return dtype

    def set_graph(self, cost_field) -> None:
        """
        Sets the field to be used for path computation

        Every value must be non-negative: shortest path search relies on non-negative arc
        costs, and a negative entry would silently return a wrong path rather than fail.
        ``+inf`` is allowed and marks an unusable link; ``NaN`` values are coerced to ``+inf``
        with a warning; negative values and ``-inf`` are rejected.

        :Arguments:
            **cost_field** (:obj:`str`): Field name. Must be numeric and non-negative
        """

        cost_field = cost_field.lower()
        if cost_field not in self.graph.columns:
            raise ValueError(
                f"Field '{cost_field}' not found in graph columns. Available fields: {list(self.graph.columns)}"
            )

        if not self.graph.empty:
            raw_costs = self.graph[cost_field].to_numpy(np.float64, copy=True)
            # A negative or -inf cost breaks the assumption every shortest path routine here
            # rests on, and there is no reading of it that produces a usable graph.
            if np.isneginf(raw_costs).any():
                raise ValueError(f"Cost field '{cost_field}' contains -inf values.")
            if (raw_costs < 0).any():
                raise ValueError(f"Cost field '{cost_field}' contains negative values.")
            # NaN is different: real networks carry it for links with no data for this field -
            # the Coquimbo example ships a travel_time column like that. It means "unusable",
            # which +inf expresses exactly, so coerce rather than reject and say so once.
            nan_costs = np.isnan(raw_costs)
            if nan_costs.any():
                logger.warning(
                    f"Cost field '{cost_field}' has {int(nan_costs.sum())} NaN values. "
                    "They are treated as unusable links (+inf)."
                )
                raw_costs[nan_costs] = np.inf

            # Always install the validated float64 vector into self.graph before groupby
            self.graph = self.graph.copy()
            self.graph[cost_field] = raw_costs
            if (
                not self.compact_graph.empty
                and "__compressed_id__" in self.graph.columns
                and self.__graph_groupby is not None
            ):
                self.__graph_groupby = self.graph.groupby(["__compressed_id__"])

        self.cost_field = cost_field

        # Restore turn penalties from master copies (they are preserved across cost field changes).
        # Turn penalties are always applied regardless of cost field - the user is responsible
        # for ensuring unit consistency between the cost field and penalty values.
        if self._turn_penalties_master is not None:
            self.turn_penalties[:] = self._turn_penalties_master[:]
        if self._compact_turn_penalties_master is not None:
            self.compact_turn_penalties[:] = self._compact_turn_penalties_master[:]

        # We only have a compact graph if we have added centroids, as that's used for skimming and assignment
        if not self.compact_graph.empty:
            self.compact_cost = np.zeros(self.compact_graph.id.max() + 2, self.__float_type)
            if self.__graph_groupby is None or self.__graph_groupby.obj is not self.graph:
                self.__graph_groupby = self.graph.groupby(["__compressed_id__"])
            df = self.__graph_groupby[[cost_field]].sum().reset_index()
            self.compact_cost[df.index.values] = df[cost_field].values
        else:
            self.__graph_groupby = None
            self.compact_cost = np.zeros(1, self.__float_type)
            if self.skim_fields and self.compact_num_nodes > 0:
                self.compact_skims = np.zeros((1, len(self.skim_fields) + 1), self.__float_type)

        if self.graph.empty or self.num_links == 0:
            self.cost = np.zeros(0, dtype=self.__float_type)
        elif self.graph[cost_field].dtype == self.__float_type:
            self.cost = np.array(self.graph[cost_field].values, copy=True)
        else:
            self.cost = np.array(self.graph[cost_field].values, dtype=self.__float_type)
            logger.warning("Cost field with wrong type. Converting to float64")

        self.__build_derived_properties()

    def _build_crosswalk(self) -> None:
        """Maps every __supernet_id__ onto the compressed link that absorbed it."""
        if self.graph.empty or "__compressed_id__" not in self.graph.columns:
            self._crosswalk = None
            return
        supernet_ids = self.graph.__supernet_id__.to_numpy(copy=False)
        compressed_ids = self.graph.__compressed_id__.to_numpy(copy=False)
        supernet_size = int(supernet_ids.max() + 1) if supernet_ids.size > 0 else self.graph.shape[0]
        size = max(self.graph.shape[0], supernet_size)
        crosswalk = np.full(size, self.compact_num_links, dtype=self.__int_type)
        crosswalk[supernet_ids] = compressed_ids
        self._crosswalk = crosswalk

    def compact_costs_from_link_costs(self, link_costs: np.ndarray) -> None:
        """Updates compact_cost from link costs indexed by __supernet_id__.

        link_costs must be a 1D array of non-negative numeric values whose length
        matches the number of links in the graph. ``+inf`` is allowed and marks an
        unusable link; ``NaN`` values are coerced to ``+inf``; negative values and
        ``-inf`` are rejected.
        """
        if self.compact_num_links > 0:
            costs_arr = np.asarray(link_costs, dtype=self.__float_type)
            if self._crosswalk is None:
                self._build_crosswalk()
            expected_len = len(self._crosswalk) if self._crosswalk is not None else self.graph.shape[0]
            if costs_arr.shape[0] not in (self.graph.shape[0], expected_len):
                raise ValueError(
                    f"link_costs array length {costs_arr.shape[0]} does not match "
                    f"graph link count {self.graph.shape[0]}"
                )
            if np.isneginf(costs_arr).any():
                raise ValueError("link_costs contains -inf values.")
            if (costs_arr < 0).any():
                raise ValueError("link_costs contains negative values.")
            if np.isnan(costs_arr).any():
                costs_arr = costs_arr.copy()
                costs_arr[np.isnan(costs_arr)] = np.inf

            if costs_arr.shape[0] == self.graph.shape[0] and expected_len != self.graph.shape[0]:
                expanded = np.zeros(expected_len, dtype=self.__float_type)
                expanded[self.graph.__supernet_id__.to_numpy(copy=False)] = costs_arr
                costs_arr = expanded

            if len(self.compact_cost) <= 1 and not self.compact_graph.empty:
                self.compact_cost = np.zeros(self.compact_graph.id.max() + 2, self.__float_type)
            from aequilibrae.paths.cython.parallel_numpy import aggregate_link_costs

            costs_arr = np.ascontiguousarray(costs_arr, dtype=self.__float_type)
            aggregate_link_costs(costs_arr, self.compact_cost, self._crosswalk)

    def set_skimming(self, skim_fields: list) -> None:
        """
        Sets the list of skims to be computed

        Skimming with A* may produce results that differ from traditional Dijkstra's due to its use a heuristic.

        :Arguments:
            **skim_fields** (:obj:`list`): Fields must be numeric
        """
        if not skim_fields:
            self.skim_fields = []
            self.skims = np.array([])

        if isinstance(skim_fields, str):
            skim_fields = [skim_fields]
        elif not isinstance(skim_fields, list):
            raise ValueError("You need to provide a list of skims or the same of a single field")
        skim_fields = [skim.lower() for skim in skim_fields]

        # Check if list of fields make sense
        k = [x for x in skim_fields if x not in self.graph.columns]
        if k:
            raise ValueError("At least one of the skim fields does not exist in the graph: {}".format(",".join(k)))

        try:
            for field in skim_fields:
                self.graph[field] = pd.to_numeric(self.graph[field], errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Skim fields must contain numeric values: {exc}") from exc

        if self.num_links == 0:
            self.skims = np.zeros((1, len(skim_fields) + 1), self.__float_type)
            if self.centroids is not None and self.centroids.shape[0]:
                self.compact_skims = np.zeros((1, len(skim_fields) + 1), self.__float_type)
            self.skim_fields = skim_fields
            return

        if self.centroids is not None and self.centroids.shape[0]:
            self.compact_skims = np.zeros((self.compact_num_links + 1, len(skim_fields) + 1), self.__float_type)

            gpb = self.__graph_groupby
            if gpb.obj is not self.graph or any(x not in gpb.obj.columns for x in skim_fields):
                gpb = self.graph.groupby(["__compressed_id__"])
                self.__graph_groupby = gpb

            df = gpb[skim_fields].sum().reset_index()

            for i, skm in enumerate(skim_fields):
                self.compact_skims[df.index.values, i] = df[skm].values.astype(self.__float_type)

        self.skims = np.zeros((self.num_links, len(skim_fields) + 1), self.__float_type)
        t = [x for x in skim_fields if self.graph[x].dtype != self.__float_type]
        if t:
            Warning("Some skim field with wrong type. Converting to float64")
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].astype(self.__float_type).values[:]
        else:
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].values[:]
        self.skim_fields = skim_fields

    def set_blocked_centroid_flows(self, block_centroid_flows) -> None:
        """
        Chooses whether paths are allowed to pass through centroid connector turns.

        With explicit turn restrictions, turn restrictions between centroid connectors are inserted.

        Default value is ``True``.

        :Arguments:
            **block_centroid_flows** (:obj:`bool`): Whether to block flows through centroids.
        """
        if not isinstance(block_centroid_flows, bool):
            raise TypeError("block_centroid_flows needs to be boolean")
        if self.num_zones == 0:
            logger.warning("No centroids in the model. Nothing to block")
            return
        if self.block_centroid_flows != block_centroid_flows:
            self.block_centroid_flows = block_centroid_flows
            # Turn-based routing uses connector bans; refresh them when blocking changes.
            # Node-based routing reads the blocking flag from the prepared context.
            self._build_turn_csr_structures()
            self._id = uuid.uuid4().hex

    @property
    def has_turn_restrictions(self) -> bool:
        """Returns True if the graph has turn restrictions loaded and active."""
        return self._has_turn_restrictions

    @property
    def allow_uturns_everywhere(self) -> bool:
        """Returns the allow U-turns everywhere setting that was used to build the graph."""
        return self._allow_uturns_everywhere

    @property
    def allow_path_uturns(self) -> bool:
        """Returns the current allow path U-turn permission setting."""
        return self._allow_path_uturns

    @staticmethod
    def _normalise_turn_penalty(penalty: object) -> float:
        if penalty is None or pd.isna(penalty):
            return np.inf

        try:
            value = float(penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Turn penalties must be None/NaN/+inf for a prohibition or a non-negative finite cost"
            ) from exc

        if value < 0:
            raise ValueError("Negative turn penalties are not allowed. Use None/NaN/+inf for a prohibition")
        elif not np.isfinite(value):
            return float("inf")
        return value

    def _compute_effective_turn_vias(self) -> set[int]:
        if self._turn_restrictions is None or len(self._turn_restrictions) == 0:
            return set()
        if self.graph.empty:
            return {int(v) for v in self._turn_restrictions["via_node"].dropna().unique()}

        # _graph_generation is bumped wherever self.graph is replaced (prepare_graph and
        # _initialize_empty_topology are the only two), so it already separates same-shaped
        # topologies. A summed-node-id fingerprint would not: it is permutation invariant and
        # blind to swapping one edge for another with the same endpoint total.
        cache_key = (
            self._turn_restrictions_generation,
            self._graph_generation,
            len(self._turn_restrictions),
            self.graph.shape[0],
        )
        cached = self._effective_vias_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        tr = self._turn_restrictions
        fn = pd.to_numeric(tr["from_node"], errors="coerce").to_numpy(np.float64)
        vn = pd.to_numeric(tr["via_node"], errors="coerce").to_numpy(np.float64)
        tn = pd.to_numeric(tr["to_node"], errors="coerce").to_numpy(np.float64)

        max_id = self.nodes_to_indices.shape[0] - 1
        usable = ~(np.isnan(fn) | np.isnan(vn) | np.isnan(tn))
        usable &= (fn >= 0) & (fn <= max_id) & (vn >= 0) & (vn <= max_id) & (tn >= 0) & (tn <= max_id)
        if not usable.any():
            return set()

        # np.where evaluates both branches, so clip before gathering rather than after.
        safe = np.clip(np.nan_to_num(np.stack([fn, vn, tn]), nan=0.0), 0, max_id).astype(np.int64)
        f_idx, v_idx, t_idx = self.nodes_to_indices[safe]
        usable &= (f_idx >= 0) & (v_idx >= 0) & (t_idx >= 0)
        if not usable.any():
            return set()

        # Pack each (tail, head) pair into a single int64 so membership is one vectorised
        # lookup instead of a Python set over every directed arc. Working in node-index
        # space keeps the stride at num_nodes, well clear of int64 overflow.
        stride = np.int64(self.num_nodes)
        edge_keys = np.unique(
            self.graph.a_node.to_numpy(np.int64, copy=False) * stride + self.graph.b_node.to_numpy(np.int64, copy=False)
        )
        has_incoming = np.isin(f_idx * stride + v_idx, edge_keys)
        has_outgoing = np.isin(v_idx * stride + t_idx, edge_keys)

        effective = usable & has_incoming & has_outgoing
        if effective.sum() < usable.sum():
            logger.warning("Turn restrictions contain movements between non-adjacent links that will have no effect")

        result = {int(v) for v in vn[effective]}
        self._effective_vias_cache = (cache_key, result)
        return result

    def _compute_turn_topology_signature(self) -> tuple:
        via_nodes: tuple[int, ...] = tuple(sorted(self._compute_effective_turn_vias()))
        return (via_nodes, bool(self._allow_path_uturns), bool(self._allow_uturns_everywhere))

    def set_turn_restrictions(self, turn_restrictions: pd.DataFrame, allow_path_uturns: bool = False) -> None:
        """
        Sets turn restrictions for the graph. When turn restrictions are set,
        the graph will use arc-based Dijkstra for path computation.

        Turn penalties are in the same time unit as the graph's cost field.

        :Arguments:
            **turn_restrictions** (:obj:`pd.DataFrame`): DataFrame with columns:
                - from_node: incoming leg origin node
                - via_node: turn node
                - to_node: outgoing leg destination node
                - penalty: turn penalty in time units

                ``None``, ``NaN`` and ``+inf`` are treated as prohibited turns.
                Negative values are not allowed.

                Restrictions are interpreted as directed movement sequences
                ``from_node -> via_node -> to_node``.

            **allow_path_uturns** (:obj:`bool`): Whether U-turns are allowed within paths.
                U-turns are node-based transitions that return to the tail node of the
                current directed arc. Default is ``False``.
        """
        if turn_restrictions is None:
            self._turn_restrictions = None
            self._allow_path_uturns = allow_path_uturns
            self._turn_restrictions_generation += 1
            self._effective_vias_cache = None
            if not self.graph.empty and self.num_nodes >= 0:
                if self.centroids is not None and self.centroids.shape[0] > 0:
                    self._reprepare(self.centroids)
                else:
                    self._turn_topology_signature = self._compute_turn_topology_signature()
                    self._build_turn_csr_structures()
                    self._id = uuid.uuid4().hex
            return

        # An empty table still carries policy: ``allow_path_uturns`` is a global setting and
        # must be honoured whether or not this traffic class has any applicable restriction.
        # The table is still installed (as an empty one), so ``has_turn_restrictions`` stays
        # False and the node-based kernel is still selected.
        required_cols = {"from_node", "via_node", "to_node", "penalty"}
        if not required_cols.issubset(turn_restrictions.columns):
            missing = required_cols - set(turn_restrictions.columns)
            raise ValueError(
                f"Turn restrictions table missing required columns: {missing}. "
                "Run project.upgrade() to update the schema."
            )

        normalised = turn_restrictions.copy()
        normalised["from_node"] = normalised["from_node"].astype(np.int64)
        normalised["via_node"] = normalised["via_node"].astype(np.int64)
        normalised["to_node"] = normalised["to_node"].astype(np.int64)
        normalised["penalty"] = normalised["penalty"].apply(self._normalise_turn_penalty)

        mvmt_cols = ["from_node", "via_node", "to_node"]
        if not normalised.empty and normalised.duplicated(subset=mvmt_cols).any():
            # Resolve each repeated movement in place: a prohibition anywhere in the group wins,
            # and finite penalties that disagree are an input error rather than a silent pick.
            # Keep the first row of each group so the caller's other columns - modes,
            # restriction_id, geometry - survive. Rebuilding the frame from the four required
            # columns would make the schema depend on whether duplicates happened to exist.
            resolved = []
            for mvmt, group in normalised.groupby(mvmt_cols, sort=False):
                pens = group["penalty"].to_numpy(dtype=np.float64)
                finite = pens[np.isfinite(pens)]
                if np.any(~np.isfinite(pens)):
                    final_pen = np.inf
                elif len(finite) > 1 and not np.all(finite == finite[0]):
                    raise ValueError(
                        f"Conflicting duplicate turn penalties for movement {mvmt}: {sorted(set(finite.tolist()))}"
                    )
                else:
                    final_pen = finite[0] if len(finite) else np.inf
                row = group.iloc[[0]].copy()
                row["penalty"] = final_pen
                resolved.append(row)
            normalised = pd.concat(resolved, ignore_index=True)

        self._turn_restrictions = normalised
        self._allow_path_uturns = allow_path_uturns
        self._turn_restrictions_generation += 1

        if self.graph.empty or self.num_nodes < 0:
            return

        new_sig = self._compute_turn_topology_signature()
        if new_sig != self._turn_topology_signature:
            if self.centroids is not None and self.centroids.shape[0] > 0:
                self._reprepare(self.centroids)
            else:
                self._turn_topology_signature = new_sig
                self._build_turn_csr_structures()
                self._id = uuid.uuid4().hex
        else:
            self._build_turn_csr_structures()
            self._id = uuid.uuid4().hex
        logger.info(f"Set {len(turn_restrictions)} turn restrictions, {allow_path_uturns=}")

    def clear_turn_restrictions(self) -> None:
        """Clears all turn restrictions from the graph."""
        self._turn_restrictions = None
        self._allow_path_uturns = False
        self._turn_restrictions_generation += 1
        self._effective_vias_cache = None
        if self.graph.empty or self.num_nodes < 0:
            return

        if self.centroids is not None and self.centroids.shape[0] > 0:
            self._reprepare(self.centroids)
        else:
            self._turn_topology_signature = self._compute_turn_topology_signature()
            self._build_turn_csr_structures()
            self._id = uuid.uuid4().hex
        logger.info("Cleared turn restrictions")

    def _has_multi_connector_centroid(self) -> bool:
        """
        Returns True when at least one centroid has more than one connector link.
        """
        if not self.block_centroid_flows or self.centroids is None or self.centroids.shape[0] == 0:
            return False

        centroids = self.centroids.astype(np.int64, copy=False)
        links = self.network[["link_id", "a_node", "b_node"]].copy()

        a_hits = links["a_node"].isin(centroids)
        b_hits = links["b_node"].isin(centroids)
        if not (a_hits.any() or b_hits.any()):
            return False

        centroid_rows = pd.concat(
            [
                links.loc[a_hits, ["link_id", "a_node"]].rename(columns={"a_node": "centroid"}),
                links.loc[b_hits, ["link_id", "b_node"]].rename(columns={"b_node": "centroid"}),
            ],
            axis=0,
            ignore_index=True,
        )

        connector_count = (
            centroid_rows.drop_duplicates(["centroid", "link_id"]).groupby("centroid")["link_id"].nunique()
        )
        return bool((connector_count > 1).any())

    @staticmethod
    def _generate_centroid_connector_turn_bans(
        num_zones: int,
        fs: np.ndarray,
        a_nodes_by_id: np.ndarray,
        b_nodes_by_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates prohibited turns through centroid nodes.
        """
        if num_zones <= 0 or fs.size == 0 or a_nodes_by_id.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        connector_mask = (a_nodes_by_id < num_zones) | (b_nodes_by_id < num_zones)
        connector_arcs = np.flatnonzero(connector_mask)
        if connector_arcs.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        prohibited_pairs = set()
        for from_arc in connector_arcs:
            node = int(b_nodes_by_id[from_arc])
            if node < 0 or node >= num_zones or node + 1 >= fs.shape[0]:
                continue

            for to_arc in range(int(fs[node]), int(fs[node + 1])):
                if not connector_mask[to_arc]:
                    continue
                prohibited_pairs.add((int(from_arc), int(to_arc)))

        if not prohibited_pairs:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        ordered = sorted(prohibited_pairs)
        from_arcs = np.fromiter((p[0] for p in ordered), dtype=np.int64, count=len(ordered))
        to_arcs = np.fromiter((p[1] for p in ordered), dtype=np.int64, count=len(ordered))
        penalties = np.full(len(ordered), np.inf, dtype=np.float64)
        return from_arcs, to_arcs, penalties

    def _build_turn_csr_structures(self) -> None:
        """
        Builds CSR structures for turn transitions in the full and compact graphs.
        """
        has_explicit_turns = self._turn_restrictions is not None and len(self._turn_restrictions) > 0
        if has_explicit_turns and self.nodes_to_indices.size > 0:
            tr = self._turn_restrictions
            tr_from_node = tr["from_node"].to_numpy(np.int64, copy=False)
            tr_via_node = tr["via_node"].to_numpy(np.int64, copy=False)
            tr_to_node = tr["to_node"].to_numpy(np.int64, copy=False)
            tr_penalty = pd.to_numeric(tr["penalty"], errors="coerce").to_numpy(np.float64, copy=False)

            max_idx = self.nodes_to_indices.shape[0] - 1
            valid_mask = (
                (tr_from_node >= 0)
                & (tr_via_node >= 0)
                & (tr_to_node >= 0)
                & (tr_from_node <= max_idx)
                & (tr_via_node <= max_idx)
                & (tr_to_node <= max_idx)
            )

            fn_valid = tr_from_node[valid_mask]
            vn_valid = tr_via_node[valid_mask]
            tn_valid = tr_to_node[valid_mask]
            pen_valid = tr_penalty[valid_mask]

            fn_idx = self.nodes_to_indices[fn_valid]
            vn_idx = self.nodes_to_indices[vn_valid]
            tn_idx = self.nodes_to_indices[tn_valid]

            present_mask = (fn_idx >= 0) & (vn_idx >= 0) & (tn_idx >= 0)
            tr_from_node_full = fn_idx[present_mask]
            tr_via_node_full = vn_idx[present_mask]
            tr_to_node_full = tn_idx[present_mask]
            tr_penalty_filtered = pen_valid[present_mask]
        else:
            tr_from_node_full = np.empty(0, dtype=np.int64)
            tr_via_node_full = np.empty(0, dtype=np.int64)
            tr_to_node_full = np.empty(0, dtype=np.int64)
            tr_penalty_filtered = np.empty(0, dtype=np.float64)

        num_links = self.num_links
        graph_ids = self.graph["id"].to_numpy(np.int64, copy=False)
        graph_a_by_id = np.empty(num_links, dtype=np.int64)
        graph_b_by_id = np.empty(num_links, dtype=np.int64)
        graph_a_by_id[graph_ids] = self.graph["a_node"].to_numpy(np.int64, copy=False)
        graph_b_by_id[graph_ids] = self.graph["b_node"].to_numpy(np.int64, copy=False)

        # 1) Map explicit user restrictions to directed full-graph arc IDs.
        if tr_from_node_full.size > 0:
            full_from_arcs, full_to_arcs, full_penalties = self._map_node_turn_restrictions_to_arcs(
                self.fs,
                graph_a_by_id,
                graph_b_by_id,
                tr_from_node_full,
                tr_via_node_full,
                tr_to_node_full,
                tr_penalty_filtered,
            )
        else:
            full_from_arcs = np.empty(0, dtype=np.int64)
            full_to_arcs = np.empty(0, dtype=np.int64)
            full_penalties = np.empty(0, dtype=np.float64)

        # 2) Only use connector bans with explicit turns, otherwise keep node-based centroid blocking.
        # A single bidirectional connector needs a ban only if path U-turns are allowed.
        block_centroid_turns = (
            has_explicit_turns
            and self.block_centroid_flows
            and (self._has_multi_connector_centroid() or self._allow_path_uturns)
        )
        if block_centroid_turns:
            auto_from_arcs, auto_to_arcs, auto_penalties = self._generate_centroid_connector_turn_bans(
                self.num_zones,
                self.fs,
                graph_a_by_id,
                graph_b_by_id,
            )
            if auto_from_arcs.size > 0:
                full_from_arcs = np.concatenate([full_from_arcs, auto_from_arcs])
                full_to_arcs = np.concatenate([full_to_arcs, auto_to_arcs])
                full_penalties = np.concatenate([full_penalties, auto_penalties])

        # 3) Build sparse explicit-turn CSR. Default transitions are implicit
        #    and resolved in the arc-based shortest-path kernel.
        turn_fs, turn_to_arcs, turn_penalties = self._build_sparse_turn_restriction_csr(
            self.num_links,
            full_from_arcs,
            full_to_arcs,
            full_penalties,
        )

        self.turn_fs = np.asarray(turn_fs, dtype=self.default_types("int"))
        self.turn_to_arcs = np.asarray(turn_to_arcs, dtype=self.default_types("int"))
        self.turn_penalties = np.asarray(turn_penalties, dtype=self.default_types("float"))
        self._turn_penalties_master = np.array(self.turn_penalties, copy=True)
        self.stateful, self.rep_arc = self._compute_stateful_and_rep_arc(
            self.num_nodes,
            self.num_links,
            self.fs,
            graph_b_by_id,
            graph_a_by_id,
            self.turn_fs,
        )

        if self.compact_graph.empty or self.compact_num_links <= 0:
            self.compact_turn_fs = np.zeros(1, dtype=self.default_types("int"))
            self.compact_turn_to_arcs = np.empty(0, dtype=self.default_types("int"))
            self.compact_turn_penalties = np.empty(0, dtype=self.default_types("float"))
            self._compact_turn_penalties_master = np.array(self.compact_turn_penalties, copy=True)
            self.compact_stateful = np.empty(0, dtype=np.uint8)
            self.compact_rep_arc = np.empty(0, dtype=np.int64)
            self._has_turn_restrictions = self.turn_penalties.size > 0
            return

        num_compact_links = self.compact_num_links
        compact_ids = self.compact_graph["id"].to_numpy(np.int64, copy=False)
        compact_a_by_id = np.empty(num_compact_links, dtype=np.int64)
        compact_b_by_id = np.empty(num_compact_links, dtype=np.int64)
        compact_a_by_id[compact_ids] = self.compact_graph["a_node"].to_numpy(np.int64, copy=False)
        compact_b_by_id[compact_ids] = self.compact_graph["b_node"].to_numpy(np.int64, copy=False)

        # 3) Compact turn mapping using boundary contexts (_compact_first_node and _compact_last_node)
        if tr_from_node_full.size > 0 and self._compact_first_node.size > 0 and self.compact_all_nodes.size > 0:
            compact_from_arcs, compact_to_arcs, compact_penalties = self._map_compact_turn_restrictions_to_arcs(
                self.compact_num_nodes,
                compact_a_by_id,
                compact_b_by_id,
                self._compact_first_node,
                self._compact_last_node,
                self.compact_all_nodes,
                self.nodes_to_indices,
                tr_from_node_full,
                tr_via_node_full,
                tr_to_node_full,
                tr_penalty_filtered,
            )
        else:
            compact_from_arcs = np.empty(0, dtype=np.int64)
            compact_to_arcs = np.empty(0, dtype=np.int64)
            compact_penalties = np.empty(0, dtype=np.float64)

        if block_centroid_turns:
            auto_c_from_arcs, auto_c_to_arcs, auto_c_penalties = self._generate_centroid_connector_turn_bans(
                self.num_zones,
                self.compact_fs,
                compact_a_by_id,
                compact_b_by_id,
            )
            if auto_c_from_arcs.size > 0:
                compact_from_arcs = np.concatenate([compact_from_arcs, auto_c_from_arcs])
                compact_to_arcs = np.concatenate([compact_to_arcs, auto_c_to_arcs])
                compact_penalties = np.concatenate([compact_penalties, auto_c_penalties])

        compact_turn_fs, compact_turn_to_arcs, compact_turn_penalties = self._build_sparse_turn_restriction_csr(
            self.compact_num_links,
            compact_from_arcs,
            compact_to_arcs,
            compact_penalties,
        )

        self.compact_turn_fs = np.asarray(compact_turn_fs, dtype=self.default_types("int"))
        self.compact_turn_to_arcs = np.asarray(compact_turn_to_arcs, dtype=self.default_types("int"))
        self.compact_turn_penalties = np.asarray(compact_turn_penalties, dtype=self.default_types("float"))
        self._compact_turn_penalties_master = np.array(self.compact_turn_penalties, copy=True)
        self.compact_stateful, self.compact_rep_arc = self._compute_stateful_and_rep_arc(
            self.compact_num_nodes,
            self.compact_num_links,
            self.compact_fs,
            compact_b_by_id,
            compact_a_by_id,
            self.compact_turn_fs,
        )

        self._has_turn_restrictions = self.turn_penalties.size > 0 or self.compact_turn_penalties.size > 0

    @staticmethod
    def _compute_stateful_and_rep_arc(
        num_nodes: int,
        num_arcs: int,
        graph_fs: np.ndarray,
        csr_indices: np.ndarray,
        a_nodes: np.ndarray,
        turn_fs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes stateful boolean array and representative arc index per node for hybrid Dijkstra.

        A node is stateful when the arc a path arrived on can still change what the path may do
        next: either the node itself carries an explicit turn entry keyed on an incoming arc (a
        via node), or it neighbours one, where a prohibition can force a reversal. Everywhere
        else the kernel keeps a single label per node, collapsed onto ``rep_arc``. See
        ``path_finding_hybrid`` for why that collapse preserves optimality.
        """
        rep_arc = np.zeros(max(num_nodes, 0), dtype=np.int64)
        stateful = np.zeros(max(num_nodes, 0), dtype=np.uint8)
        if num_nodes <= 0 or num_arcs <= 0:
            return stateful, rep_arc

        heads = csr_indices[:num_arcs]
        # The first arc entering a node is the label every plain arrival collapses onto.
        # Nodes with no incoming arc keep arc 0; nothing ever reads their label.
        reached, first_arc = np.unique(heads, return_index=True)
        rep_arc[reached] = first_arc

        if turn_fs.shape[0] <= 1:
            return stateful, rep_arc

        max_arc = min(num_arcs, turn_fs.shape[0] - 1)
        restricted_arcs = np.flatnonzero(turn_fs[1 : max_arc + 1] > turn_fs[:max_arc])
        if restricted_arcs.size == 0:
            return stateful, rep_arc

        via_nodes = np.unique(heads[restricted_arcs])
        stateful[via_nodes] = 1
        # Both neighbourhoods of a via node are stateful too: a prohibition there can force a
        # reversal, which is the one thing a collapsed label cannot express.
        out_arcs = np.concatenate([np.arange(graph_fs[v], graph_fs[v + 1], dtype=np.int64) for v in via_nodes])
        stateful[heads[out_arcs]] = 1
        stateful[a_nodes[:num_arcs][np.isin(heads, via_nodes)]] = 1

        return stateful, rep_arc

    @staticmethod
    def _map_compact_turn_restrictions_to_arcs(
        compact_num_nodes: int,
        compact_a_by_id: np.ndarray,
        compact_b_by_id: np.ndarray,
        compact_first_node: np.ndarray,
        compact_last_node: np.ndarray,
        compact_all_nodes: np.ndarray,
        nodes_to_indices: np.ndarray,
        tr_from_node_full: np.ndarray,
        tr_via_node_full: np.ndarray,
        tr_to_node_full: np.ndarray,
        tr_penalties: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Maps node-sequence restrictions to compact arc pairs using boundary contexts.
        """
        num_compact_arcs = compact_a_by_id.shape[0]
        if num_compact_arcs == 0 or tr_from_node_full.shape[0] == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        # compact_all_nodes holds node IDs of the full graph, so nodes_to_indices maps them all.
        nodes = compact_all_nodes[: int(compact_num_nodes)].astype(np.int64, copy=False)
        full_idx = nodes_to_indices[nodes]

        max_full = int(full_idx.max()) if full_idx.size else 0
        full_to_compact = np.full(max_full + 1, -1, dtype=np.int64)
        full_to_compact[full_idx] = np.arange(full_idx.shape[0], dtype=np.int64)

        stride = (
            int(
                max(
                    int(compact_a_by_id.max()) if compact_a_by_id.size else 0,
                    int(compact_b_by_id.max()) if compact_b_by_id.size else 0,
                    int(compact_first_node.max()) if compact_first_node.size else 0,
                    int(compact_last_node.max()) if compact_last_node.size else 0,
                    0,
                )
            )
            + 1
        )
        in_keys, in_order = GraphBase._pair_index(compact_b_by_id, compact_last_node, stride)
        out_keys, out_order = GraphBase._pair_index(compact_a_by_id, compact_first_node, stride)

        fn = tr_from_node_full.astype(np.int64, copy=False)
        vn = tr_via_node_full.astype(np.int64, copy=False)
        tn = tr_to_node_full.astype(np.int64, copy=False)
        pen = tr_penalties.astype(np.float64, copy=False)

        c_via = np.full(vn.shape[0], -1, dtype=np.int64)
        vn_in_range = (vn >= 0) & (vn <= max_full)
        c_via[vn_in_range] = full_to_compact[vn[vn_in_range]]

        valid = (c_via >= 0) & (fn >= 0) & (fn < stride) & (tn >= 0) & (tn < stride) & (c_via < stride)

        in_query = np.where(valid, c_via * np.int64(stride) + fn, np.int64(-1))
        out_query = np.where(valid, c_via * np.int64(stride) + tn, np.int64(-1))

        lo_in = np.searchsorted(in_keys, in_query, side="left")
        hi_in = np.searchsorted(in_keys, in_query, side="right")
        lo_out = np.searchsorted(out_keys, out_query, side="left")
        hi_out = np.searchsorted(out_keys, out_query, side="right")

        cin = np.where(valid, hi_in - lo_in, 0)
        cout = np.where(valid, hi_out - lo_out, 0)

        out_from_parts = []
        out_to_parts = []
        out_pen_parts = []

        single = (cin == 1) & (cout == 1)
        if np.any(single):
            out_from_parts.append(in_order[lo_in[single]])
            out_to_parts.append(out_order[lo_out[single]])
            out_pen_parts.append(pen[single])

        multi_indices = np.flatnonzero((cin > 0) & (cout > 0) & ((cin > 1) | (cout > 1)))
        if multi_indices.size > 0:
            m_from = []
            m_to = []
            m_pen = []
            for idx in multi_indices:
                p = float(pen[idx])
                for f_arc in in_order[lo_in[idx] : hi_in[idx]]:
                    for t_arc in out_order[lo_out[idx] : hi_out[idx]]:
                        m_from.append(int(f_arc))
                        m_to.append(int(t_arc))
                        m_pen.append(p)
            out_from_parts.append(np.asarray(m_from, dtype=np.int64))
            out_to_parts.append(np.asarray(m_to, dtype=np.int64))
            out_pen_parts.append(np.asarray(m_pen, dtype=np.float64))

        if not out_from_parts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        res_from = np.concatenate(out_from_parts) if len(out_from_parts) > 1 else out_from_parts[0]
        res_to = np.concatenate(out_to_parts) if len(out_to_parts) > 1 else out_to_parts[0]
        res_pen = np.concatenate(out_pen_parts) if len(out_pen_parts) > 1 else out_pen_parts[0]

        return res_from, res_to, res_pen

    @staticmethod
    def _pair_index(left: np.ndarray, right: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """Builds a sorted (packed_key, arc_id) index so pair lookups are binary searches.

        Packing into one int64 keeps the whole index in NumPy; both components are node
        indices below ``stride``, so the largest key stays well inside int64.
        """
        keys = left.astype(np.int64, copy=False) * np.int64(stride) + right.astype(np.int64, copy=False)
        order = np.argsort(keys, kind="stable")
        return keys[order], order

    @staticmethod
    def _pair_lookup(sorted_keys: np.ndarray, order: np.ndarray, left: int, right: int, stride: int) -> np.ndarray:
        """Returns the arc IDs registered for one (left, right) pair, in ascending arc order."""
        key = np.int64(left) * np.int64(stride) + np.int64(right)
        lo = int(np.searchsorted(sorted_keys, key, side="left"))
        hi = int(np.searchsorted(sorted_keys, key, side="right"))
        return order[lo:hi]

    @staticmethod
    def _map_node_turn_restrictions_to_arcs(
        fs: np.ndarray,
        a_by_arc: np.ndarray,
        b_by_arc: np.ndarray,
        from_nodes: np.ndarray,
        via_nodes: np.ndarray,
        to_nodes: np.ndarray,
        penalties: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maps node-sequence restrictions (from_node, via_node, to_node) to directed arc pairs."""
        num_arcs = a_by_arc.shape[0]
        if num_arcs == 0 or from_nodes.shape[0] == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        max_a = int(a_by_arc.max()) if a_by_arc.size else 0
        max_b = int(b_by_arc.max()) if b_by_arc.size else 0
        stride = int(max(max_a, max_b, 0)) + 1
        sorted_keys, order = GraphBase._pair_index(a_by_arc, b_by_arc, stride)

        fn = from_nodes.astype(np.int64, copy=False)
        vn = via_nodes.astype(np.int64, copy=False)
        tn = to_nodes.astype(np.int64, copy=False)
        pen = penalties.astype(np.float64, copy=False)

        valid = (fn >= 0) & (fn < stride) & (vn >= 0) & (vn < stride) & (tn >= 0) & (tn < stride)

        in_query = np.where(valid, fn * np.int64(stride) + vn, np.int64(-1))
        out_query = np.where(valid, vn * np.int64(stride) + tn, np.int64(-1))

        lo_in = np.searchsorted(sorted_keys, in_query, side="left")
        hi_in = np.searchsorted(sorted_keys, in_query, side="right")
        lo_out = np.searchsorted(sorted_keys, out_query, side="left")
        hi_out = np.searchsorted(sorted_keys, out_query, side="right")

        cin = np.where(valid, hi_in - lo_in, 0)
        cout = np.where(valid, hi_out - lo_out, 0)

        out_from_parts = []
        out_to_parts = []
        out_pen_parts = []

        single = (cin == 1) & (cout == 1)
        if np.any(single):
            f_arcs = order[lo_in[single]]
            t_arcs = order[lo_out[single]]
            heads = b_by_arc[f_arcs]
            in_fs = (heads >= 0) & (heads + 1 < fs.shape[0])
            # Gather the forward star with the out-of-range heads clamped to a safe slot so
            # every intermediate stays the same length as `single`; `in_fs` then masks the
            # clamped rows out. Indexing `fs` with `heads[in_fs]` instead would produce a
            # shorter array and either broadcast silently or raise.
            safe_heads = np.where(in_fs, heads, 0)
            lo_fs = fs[safe_heads]
            hi_fs = fs[safe_heads + 1]
            t_valid = in_fs & (lo_fs <= t_arcs) & (t_arcs < hi_fs)
            if np.any(t_valid):
                out_from_parts.append(f_arcs[t_valid])
                out_to_parts.append(t_arcs[t_valid])
                out_pen_parts.append(pen[single][t_valid])

        multi_indices = np.flatnonzero((cin > 0) & (cout > 0) & ((cin > 1) | (cout > 1)))
        if multi_indices.size > 0:
            m_from = []
            m_to = []
            m_pen = []
            for idx in multi_indices:
                p = float(pen[idx])
                for from_arc in order[lo_in[idx] : hi_in[idx]]:
                    head = int(b_by_arc[from_arc])
                    if head < 0 or head + 1 >= fs.shape[0]:
                        continue
                    lo, hi = int(fs[head]), int(fs[head + 1])
                    for to_arc in order[lo_out[idx] : hi_out[idx]]:
                        if lo <= int(to_arc) < hi:
                            m_from.append(int(from_arc))
                            m_to.append(int(to_arc))
                            m_pen.append(p)
            out_from_parts.append(np.asarray(m_from, dtype=np.int64))
            out_to_parts.append(np.asarray(m_to, dtype=np.int64))
            out_pen_parts.append(np.asarray(m_pen, dtype=np.float64))

        if not out_from_parts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        res_from = np.concatenate(out_from_parts) if len(out_from_parts) > 1 else out_from_parts[0]
        res_to = np.concatenate(out_to_parts) if len(out_to_parts) > 1 else out_to_parts[0]
        res_pen = np.concatenate(out_pen_parts) if len(out_pen_parts) > 1 else out_pen_parts[0]

        return res_from, res_to, res_pen

    @staticmethod
    def _build_sparse_turn_restriction_csr(
        num_arcs: int,
        restricted_from_arcs: np.ndarray,
        restricted_to_arcs: np.ndarray,
        restricted_penalties: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a sparse CSR of explicit turn restrictions/penalties only.

        U-turn handling and unrestricted transitions are handled at pathfinding time.
        This keeps the turn-restriction structure compact when only a small subset of
        arc pairs has custom penalties or prohibitions.
        """
        if num_arcs <= 0:
            return (
                np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        key_to_penalty: dict[tuple[int, int], float] = {}
        prohibited_keys: set[tuple[int, int]] = set()

        # First collect all prohibited keys so prohibition dominance wins unconditionally
        for f_arc, t_arc, penalty in zip(restricted_from_arcs, restricted_to_arcs, restricted_penalties, strict=True):
            f_arc = int(f_arc)
            t_arc = int(t_arc)
            if f_arc < 0 or t_arc < 0 or f_arc >= num_arcs or t_arc >= num_arcs:
                continue
            if penalty < 0:
                raise ValueError("Negative turn penalties are not allowed. Use None/NaN/+inf for a prohibition")
            if not np.isfinite(penalty):
                prohibited_keys.add((f_arc, t_arc))

        for f_arc, t_arc, penalty in zip(restricted_from_arcs, restricted_to_arcs, restricted_penalties, strict=True):
            f_arc = int(f_arc)
            t_arc = int(t_arc)
            if f_arc < 0 or t_arc < 0 or f_arc >= num_arcs or t_arc >= num_arcs:
                continue
            key = (f_arc, t_arc)
            if key in prohibited_keys:
                continue
            if key in key_to_penalty:
                if key_to_penalty[key] != float(penalty):
                    raise ValueError(
                        f"Conflicting duplicate turn penalties for arc pair {key}: "
                        f"{key_to_penalty[key]} vs {float(penalty)}"
                    )
            else:
                key_to_penalty[key] = float(penalty)

        if not prohibited_keys and not key_to_penalty:
            return (
                np.zeros(num_arcs + 1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        # Keep entries sorted by (from_arc, to_arc) to support predictable
        # linear scans + early-break in the Cython kernel.
        entries = [(f, t, np.inf) for (f, t) in prohibited_keys]
        entries.extend((f, t, p) for (f, t), p in key_to_penalty.items())
        entries.sort(key=lambda x: (x[0], x[1]))

        turn_fs = np.zeros(num_arcs + 1, dtype=np.int64)
        turn_to_arcs = np.empty(len(entries), dtype=np.int64)
        turn_penalties = np.empty(len(entries), dtype=np.float64)

        for idx, (f_arc, t_arc, penalty) in enumerate(entries):
            turn_to_arcs[idx] = t_arc
            turn_penalties[idx] = penalty
            turn_fs[f_arc + 1] += 1

        np.cumsum(turn_fs, out=turn_fs)
        return turn_fs, turn_to_arcs, turn_penalties

    def set_turn_penalty_dimension(self, unit: str):
        """Sets the unit for turn penalties."""
        self.turn_penalty_dimension = unit

    def _backup_penalties(self):
        """Backs up valid turn penalties to master if not already done."""
        if self.turn_penalties.size > 0:
            self._turn_penalties_master = np.array(self.turn_penalties, copy=True)
        if self.compact_turn_penalties.size > 0:
            self._compact_turn_penalties_master = np.array(self.compact_turn_penalties, copy=True)

    # Procedure to pickle graph and save to disk
    def save_to_disk(self, filename: str) -> None:
        """
        Saves graph to disk

        :Arguments:
            **filename** (:obj:`str`): Path to file. Usual file extension is ``aeg``.
        """
        mygraph = {}
        mygraph["network"] = self.network
        mygraph["graph"] = self.graph
        mygraph["compact_graph"] = self.compact_graph
        mygraph["description"] = self.description
        mygraph["num_links"] = self.num_links
        mygraph["turn_penalty_dimension"] = self.turn_penalty_dimension
        mygraph["turn_skim_fields"] = self.turn_skim_fields
        mygraph["all_nodes"] = self.all_nodes
        mygraph["nodes_to_indices"] = self.nodes_to_indices
        mygraph["num_nodes"] = self.num_nodes
        mygraph["fs"] = self.fs
        mygraph["cost"] = self.cost
        mygraph["cost_field"] = self.cost_field
        mygraph["skims"] = self.skims
        mygraph["skim_fields"] = self.skim_fields
        mygraph["block_centroid_flows"] = self.block_centroid_flows
        mygraph["centroids"] = self.centroids
        mygraph["graph_id"] = self._id
        mygraph["mode"] = self.mode

        # Turn restrictions data
        mygraph["turn_restrictions"] = self._turn_restrictions
        mygraph["allow_uturns_everywhere"] = self._allow_uturns_everywhere
        mygraph["allow_path_uturns"] = self._allow_path_uturns
        mygraph["has_turn_restrictions"] = self._has_turn_restrictions
        mygraph["turn_fs"] = self.turn_fs
        mygraph["turn_to_arcs"] = self.turn_to_arcs
        mygraph["turn_penalties"] = (
            self._turn_penalties_master if self._turn_penalties_master is not None else self.turn_penalties
        )
        mygraph["compact_turn_fs"] = self.compact_turn_fs
        mygraph["compact_turn_to_arcs"] = self.compact_turn_to_arcs
        mygraph["compact_turn_penalties"] = (
            self._compact_turn_penalties_master
            if self._compact_turn_penalties_master is not None
            else self.compact_turn_penalties
        )

        mygraph["_remove_dead_ends"] = self._remove_dead_ends

        with open(filename, "wb") as f:
            pickle.dump(mygraph, f)

    def load_from_disk(self, filename: str) -> None:
        """
        Loads graph from disk

        :Arguments:
            **filename** (:obj:`str`): Path to file
        """
        with open(filename, "rb") as f:
            mygraph = pickle.load(f)

        self.description = mygraph.get("description", "No description added so far")
        # Graph files written before the network became a DataFrame store it as a record
        # array, and this load path re-prepares the graph from it rather than trusting
        # the saved derived arrays.
        self.network = mygraph["network"]
        if isinstance(self.network, np.ndarray):
            self.network = pd.DataFrame(self.network)
        self.mode = mygraph.get("mode", "")
        self.turn_penalty_dimension = mygraph.get("turn_penalty_dimension", "time")
        self.centroids = mygraph.get("centroids", None)
        self._remove_dead_ends = mygraph.get("_remove_dead_ends", True)
        self._allow_uturns_everywhere = mygraph.get("allow_uturns_everywhere", mygraph.get("allow_uturns", False))
        self._allow_path_uturns = mygraph.get("allow_path_uturns", False)
        self._turn_restrictions = mygraph.get("turn_restrictions", None)

        # Installed before re-preparation so _reprepare restores them from self.
        self.cost_field = mygraph.get("cost_field", False)
        self.skim_fields = mygraph.get("skim_fields", [])
        self.turn_skim_fields = mygraph.get("turn_skim_fields", [])
        self.block_centroid_flows = mygraph.get("block_centroid_flows", True)

        if self.centroids is not None and len(self.centroids) > 0:
            self._reprepare(self.centroids)
        elif not self.network.empty:
            self._reprepare(None)
        else:
            self.__build_derived_properties()

        self.__build_derived_properties()

    def __build_derived_properties(self):
        if self.centroids is None:
            return
        self.num_zones = self.centroids.shape[0] if self.centroids.shape else 0

    def available_skims(self) -> List[str]:
        """
        Returns graph fields that are available to be set as skims.

        :Returns:
            **list** (:obj:`str`): Skimmeable field names
        """
        return [x for x in self.graph.columns if x not in ["link_id", "a_node", "b_node", "direction", "id"]]

    # We check if all minimum fields are there
    def __network_error_checking__(self):
        # Checking field names
        has_fields = self.network.columns
        must_fields = ["link_id", "a_node", "b_node", "direction"]
        for field in must_fields:
            if field not in has_fields:
                raise ValueError(f"could not find field {field} in the network array")

        # Uniqueness of the id
        link_ids = self.network["link_id"].astype(int)
        if link_ids.shape[0] != np.unique(link_ids).shape[0]:
            raise ValueError('"link_id" field not unique')

            # Direction values
        if np.max(self.network["direction"]) > 1 or np.min(self.network["direction"]) < -1:
            raise ValueError('"direction" field not limited to (-1,0,1) values')

        if "id" not in self.network.columns:
            self.network = self.network.assign(id=np.nan)

    def __determine_types__(self, new_type, current_type):
        if new_type.isdigit():
            new_type = int(new_type)
        else:
            try:
                new_type = float(new_type)
            except ValueError as verr:
                logger.warning("Could not convert {} - {}".format(new_type, verr.__str__()))
        if isinstance(new_type, int):
            def_type = int
            if current_type is float:
                def_type = float
            elif current_type is str:
                def_type = str
        elif isinstance(new_type, float):
            def_type = float
            if current_type is str:
                def_type = str
        elif isinstance(new_type, str):
            def_type = str
        else:
            raise ValueError("WRONG TYPE OR NULL VALUE")
        return def_type

    def save_compressed_correspondence(self, path, mode_name, mode_id):
        """Save graph and nodes_to_indices to disk"""
        graph_path = join(path, f"correspondence_c{mode_name}_{mode_id}.feather")
        self.graph.to_feather(graph_path)
        node_path = join(path, f"nodes_to_indices_c{mode_name}_{mode_id}.feather")
        pd.DataFrame(self.nodes_to_indices, columns=["node_index"]).to_feather(node_path)

    def create_compressed_link_network_mapping(
        self,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.generic]]:
        """
        Create three arrays providing a mapping of compressed ID to link ID.

        Uses sparse compression. Index 'idx' by the by compressed ID and compressed ID + 1, the
        network IDs are then in the range ``idx[id]:idx[id + 1]``.

        Links not in the compressed graph are not contained within the 'data' array.

        'node_mapping' provides an easy way to check if a node index is present within the compressed graph. If the
        value is -1 then the node has been removed, either by compression of dead end link removal. If the value is
        greater than or equal to 0, then that value is the compressed node index.

        .. code-block:: python

            >>> project = create_example(project_path)

            >>> project.network.build_graphs()

            >>> graph = project.network.graphs['c']
            >>> graph.prepare_graph(np.arange(1,25))

            >>> idx, data, node_mapping = graph.create_compressed_link_network_mapping()

            >>> project.close()

        :Returns:
            **idx** (:obj:`np.array`): index array for ``data``

            **data** (:obj:`np.array`): array of link ids

            **node_mapping**: (:obj:`np.array`): array of node_mapping ids
        """
        return create_compressed_link_network_mapping(self)

    def __setattr__(self, key, value):
        if key == "network" and isinstance(value, pd.DataFrame):
            value.columns = [col.lower() for col in value.columns]
        super().__setattr__(key, value)


class Graph(GraphBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TransitGraph(GraphBase):
    def __init__(self, config: Optional[dict] = None, od_node_mapping: Optional[pd.DataFrame] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self.od_node_mapping = od_node_mapping
        self.mode = "t"

    @property
    def config(self):
        return self._config


def sanitise_centroids(centroids: np.ndarray) -> np.ndarray:
    if len(centroids.shape) != 1:
        raise ValueError(f"centroids must be 1D, got: {centroids.shape}")
    elif centroids.shape[0] == 0:
        raise ValueError("centroids must contain at least one value")

    centroids = centroids.astype("uint32", order="C", casting="same_value", subok=False)
    centroids = np.sort(centroids)  # Makes a copy
    if (np.diff(centroids) == 0).any():
        raise ValueError("centroids must be unique")

    return centroids  # a copy of the unique, sorted, C-contiguous, uint_32 centroids


def sanitise_network(network: pd.DataFrame) -> pd.DataFrame:
    required_fields = {"link_id", "a_node", "b_node", "direction"}
    if missing_fields := required_fields - set(network.columns):
        raise ValueError(f"network is missing the following required fields: {missing_fields}")

    network = network.copy(deep=False)  # CoW copy

    if not network["link_id"].is_unique:
        raise ValueError("'link_id' field must be unique")

    network["link_id"] = network["link_id"].to_numpy().astype("int64", order="C", casting="same_value")

    # Direction values
    if network["direction"].max() > 1 or network["direction"].min() < -1:
        raise ValueError('"direction" field not limited to (-1, 0, 1) values')

    network["direction"] = (
        network["direction"].to_numpy().astype("int8", order="C", casting="same_value")
    )  # FIXME: enforced type for direction?

    network["a_node"] = network["a_node"].to_numpy().astype("uint32", order="C", casting="same_value")
    network["b_node"] = network["b_node"].to_numpy().astype("uint32", order="C", casting="same_value")

    return network


def _make_unidirectional(network: pd.DataFrame) -> pd.DataFrame:
    """Expand links into directed edges and collapse paired _ab/_ba columns."""
    required = ["link_id", "a_node", "b_node", "direction"]
    names = required.copy()

    for column in network.columns:
        # Internal edge IDs are regenerated after node mapping and sorting.
        if column in required + ["id"] or column[:-3] in required + ["id"]:
            continue

        if column.endswith("_ab"):
            if column[:-3] + "_ba" not in network.columns:
                raise ValueError(f"Field {column} exists for ab direction but does not exist for ba")
            names.append(column[:-3])
        elif column.endswith("_ba"):
            if column[:-3] + "_ab" not in network.columns:
                raise ValueError(f"Field {column} exists for ba direction but does not exist for ab")
        else:
            names.append(column)

    edges = []
    for direction, suffix in ((1, "_ab"), (-1, "_ba")):
        columns = [name if name in network.columns else name + suffix for name in names]
        directed = network.loc[network["direction"] != -direction, columns].copy()
        directed.columns = names
        directed["direction"] = np.full(len(directed), direction, dtype=np.int8)

        if direction == -1:
            a_nodes = directed["a_node"].to_numpy(copy=True)
            directed["a_node"] = directed["b_node"].to_numpy(copy=True)
            directed["b_node"] = a_nodes

        edges.append(directed)

    return pd.concat(edges, ignore_index=True)


def _map_centroids(
    graph: pd.DataFrame,
    centroids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Map nodes to centroid-first indices, sort edges, and assign edge IDs."""
    nodes = np.unique(np.concatenate((graph["a_node"].to_numpy(), graph["b_node"].to_numpy())))

    present = np.isin(centroids, nodes, assume_unique=True)
    if not present.all():
        warnings.warn(
            "Found centroids not present in the graph!\n" + str(centroids[~present]),
            stacklevel=2,
        )

    non_centroids = np.setdiff1d(nodes, centroids, assume_unique=True)
    all_nodes = np.concatenate((centroids, non_centroids)).astype(np.uint32, casting="same_value", copy=False)

    # Signed indices retain the -1 sentinel for node IDs absent from the graph.
    nodes_to_indices = np.full(int(all_nodes.max()) + 1, -1, dtype=np.int64)
    nodes_to_indices[all_nodes] = np.arange(len(all_nodes), dtype=np.int64)

    graph = graph.copy()
    for column in ("a_node", "b_node"):
        graph[column] = nodes_to_indices[graph[column].to_numpy()].astype(np.uint32, casting="same_value")

    # We generate IDs that we KNOW will be constant across modes
    graph = graph.sort_values(by=["link_id", "direction"])
    graph["__supernet_id__"] = np.arange(graph.shape[0]).astype("uint32")

    graph = graph.sort_values(["a_node", "b_node"]).reset_index(drop=True)
    graph["id"] = np.arange(len(graph), dtype=np.int64)

    return graph, all_nodes, nodes_to_indices


def _build_forward_star(graph: pd.DataFrame, num_nodes: int) -> np.ndarray:
    """Build CSR offsets for edges already sorted by mapped a_node."""
    counts = np.bincount(graph["a_node"].to_numpy(), minlength=num_nodes)
    fs = np.empty(num_nodes + 1, dtype=np.int64)
    fs[0] = 0
    np.cumsum(counts, dtype=np.int64, out=fs[1:])
    return fs


def _ensure_graph_dtypes(graph: pd.DataFrame) -> pd.DataFrame:
    """Enforce required column dtypes without casting other columns."""
    graph = graph.copy()
    required_types = {
        "link_id": np.int64,
        "a_node": np.uint32,
        "b_node": np.uint32,
        "direction": np.int8,
        "id": np.int64,
    }

    for column, dtype in required_types.items():
        graph[column] = graph[column].to_numpy().astype(dtype, order="C", casting="same_value", copy=False)

    nans = ", ".join(column for column in graph.columns if graph[column].isna().any())
    if nans:
        logger.warning(
            "Found fields with at least one NaN value. Check your computations. Fields: %s",
            nans,
        )

    return graph


def _build_directed_graph(
    network: pd.DataFrame,
    centroids: np.ndarray,
    *,
    build_fs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Build from sanitised inputs, optionally omitting CSR construction."""
    graph = _make_unidirectional(network)
    graph, all_nodes, nodes_to_indices = _map_centroids(graph, centroids)
    graph = _ensure_graph_dtypes(graph)

    num_nodes = len(all_nodes)
    fs = _build_forward_star(graph, num_nodes) if build_fs else None

    return all_nodes, nodes_to_indices, fs, graph


class NewTransitGraph:
    def __init__(
        self,
        network: pd.DataFrame,
        centroids: np.ndarray,  # FIXME: Centroids used to be optional?
        time_field: str,
        frequency_field: str,
        od_node_mapping: pd.DataFrame,
        a_node_field: str = "a_node",
        b_node_field: str = "b_node",
        skimming_fields: list[str] | None = None,
        config: dict | None = None,
    ):
        self.network = sanitise_network(network)
        self.centroids = sanitise_centroids(centroids)
        # FIXME: Old graphs make a distinction between free_flow_time and cost? Not sure why

        self.skimming_fields = skimming_fields if skimming_fields is not None else []
        self.time_field = time_field
        self.frequency_field = frequency_field
        self.a_node_field = a_node_field
        self.b_node_field = b_node_field

        self.od_node_mapping = od_node_mapping.copy()
        o_key, d_key = ("node_id", "node_id") if len(self.od_node_mapping.columns) == 2 else ("o_node_id", "d_node_id")

        self.all_nodes, self.nodes_to_indices, _, self.graph = _build_directed_graph(
            self.network, self.centroids, build_fs=False
        )

        self.context = HyperpathGenerating(
            self.graph,
            head=self.a_node_field,
            tail=self.b_node_field,
            trav_time=self.time_field,
            freq=self.frequency_field,
            skim_cols=self.skimming_fields,
            o_vert_ids=self.od_node_mapping[o_key].to_numpy(),  # taz_id
            d_vert_ids=self.od_node_mapping[d_key].to_numpy(),  # node_id for destination in the above taz_id
            nodes_to_indices=self.nodes_to_indices,
        )

        self._config = {
            "time_field": time_field,
            "frequency_field": frequency_field,
            "a_node_field": self.a_node_field,
            "b_node_field": self.b_node_field,
            "num_links": self.num_links,
            "num_nodes": self.num_nodes,
            "num_zones": self.num_zones,
            "skimming_fields": self.skimming_fields,
        }
        if config is not None:
            self._config.update(config)

    def set_skimming_fields(self, skimming_fields: list[str] | None = None) -> None:
        """Set the fields to skim and rebuild the hyperpath context."""
        self.skimming_fields = list(skimming_fields or [])
        self.context = HyperpathGenerating(
            self.graph,
            head=self.a_node_field,
            tail=self.b_node_field,
            trav_time=self.time_field,
            freq=self.frequency_field,
            skim_cols=self.skimming_fields,
            o_vert_ids=self.od_node_mapping[
                "node_id" if len(self.od_node_mapping.columns) == 2 else "o_node_id"
            ].to_numpy(),
            d_vert_ids=self.od_node_mapping[
                "node_id" if len(self.od_node_mapping.columns) == 2 else "d_node_id"
            ].to_numpy(),
            nodes_to_indices=self.nodes_to_indices,
        )
        self._config["skimming_fields"] = self.skimming_fields

    @property
    def num_links(self) -> int:
        return len(self.graph)

    @property
    def num_nodes(self) -> int:
        return len(self.all_nodes)

    @property
    def num_zones(self) -> int:
        return len(self.centroids)

    @property
    def mode(self) -> str:
        return "t"

    @property
    def block_centroid_flows(self) -> bool:
        return False
