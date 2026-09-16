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
        g.prepare_graph(self.centroids)
        if self.cost_field:
            g.set_graph(self.cost_field)
        if self.skim_fields:
            g.set_skimming(self.skim_fields)
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
        self.__network_error_checking__()

        self._allow_uturns_everywhere = allow_uturns_everywhere

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

        properties = self._build_directed_graph(self.network, self.centroids)
        self.all_nodes, self.num_nodes, self.nodes_to_indices, self.fs, self.graph = properties

        # We generate IDs that we KNOW will be constant across modes
        self.graph.sort_values(by=["link_id", "direction"], inplace=True)
        self.graph["__supernet_id__"] = np.arange(self.graph.shape[0]).astype(self.__int_type)
        self.graph.sort_values(by=["a_node", "b_node"], inplace=True)

        self.num_links = self.graph.shape[0]
        self.__build_derived_properties()

        if self.centroids.shape[0]:
            self.__build_compressed_graph(remove_dead_ends)
            self.compact_num_links = self.compact_graph.shape[0]

        # The cache property should be recalculated when the graph has been re-prepared
        self.compressed_link_network_mapping_idx = None
        self.compressed_link_network_mapping_data = None
        self.network_compressed_node_mapping = None

        # Rebuild turn structures whenever graph topology/indexing changes.
        # This applies both explicit user turn restrictions and the automatic
        # centroid-connector bans used when centroid flows are blocked.
        self._build_turn_csr_structures()

    def __build_compressed_graph(self, remove_dead_ends):
        build_compressed_graph(self, remove_dead_ends)

        # We build a groupby to save time later
        self.__graph_groupby = self.graph.groupby(["__compressed_id__"])

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
        for i in range(num_nodes, 1, -1):
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
            This option is incompatible with turn restrictions.

            **heuristic** (:obj:`str`): heuristic to use if ``a_star`` is enabled. Default is ``None``.
        """
        from aequilibrae.paths import PathResults

        res = PathResults(self, origin, destination, early_exit, a_star, heuristic)

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
        Excludes a list of links from a graph by setting their B node equal to their A node

        :Arguments:
            **links** (:obj:`list`): List of link IDs to be excluded from the graph
        """
        filter = self.network.link_id.isin(links)
        # We check is the list makes sense in order to warn the user
        if filter.sum() != len(set(links)):
            logger.warning("At least one link does not exist in the network and therefore cannot be excluded")

        self.network.loc[filter, "b_node"] = self.network.loc[filter, "a_node"]

        if self.centroids is not None:
            self.prepare_graph(self.centroids)
            self.set_blocked_centroid_flows(self.block_centroid_flows)
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

        :Arguments:
            **cost_field** (:obj:`str`): Field name. Must be numeric
        """

        cost_field = cost_field.lower()
        if cost_field not in self.graph.columns:
            raise ValueError(
                f"Field '{cost_field}' not found in graph columns. Available fields: {list(self.graph.columns)}"
            )

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
            df = self.__graph_groupby.sum(numeric_only=True)[[cost_field]].reset_index()
            self.compact_cost[df.index.values] = df[cost_field].values

        if self.graph[cost_field].dtype == self.__float_type:
            self.cost = np.array(self.graph[cost_field].values, copy=True)
        else:
            self.cost = np.array(self.graph[cost_field].values, dtype=self.__float_type)
            logger.warning("Cost field with wrong type. Converting to float64")

        self.__build_derived_properties()

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

        if self.centroids is not None and self.centroids.shape[0]:
            self.compact_skims = np.zeros((self.compact_num_links + 1, len(skim_fields) + 1), self.__float_type)

            gpb = self.__graph_groupby
            if any(x not in self.__graph_groupby for x in skim_fields):
                gpb = self.graph.groupby(["__compressed_id__"])

            df = gpb.sum(numeric_only=True)[skim_fields].reset_index()

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

        When enabled, AequilibraE automatically creates prohibited turns between
        centroid connectors that meet at the same node (for centroids with more than
        one connector), which activates arc-based path finding under the hood.

        Default value is ``True``.

        :Arguments:
            **block_centroid_flows** (:obj:`bool`): Whether to block connector-to-connector
            flow through centroids using automatic turn prohibitions.
        """
        if not isinstance(block_centroid_flows, bool):
            raise TypeError("block_centroid_flows needs to be boolean")
        if self.num_zones == 0:
            logger.warning("No centroids in the model. Nothing to block")
            return
        self.block_centroid_flows = block_centroid_flows
        self._build_turn_csr_structures()

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
        required_cols = {"from_node", "via_node", "to_node", "penalty"}
        if not required_cols.issubset(turn_restrictions.columns):
            missing = required_cols - set(turn_restrictions.columns)
            raise ValueError(f"Turn restrictions DataFrame missing required columns: {missing}")

        normalised = turn_restrictions.copy()
        normalised["penalty"] = normalised["penalty"].apply(self._normalise_turn_penalty)

        # Keep a canonical copy of the user-provided turn table.
        # The table is later mapped to arc IDs for both full and compact graphs.
        self._turn_restrictions = normalised
        self._allow_path_uturns = allow_path_uturns
        # Rebuild turn CSR structures for the full graph (and compact graph if present)
        self._build_turn_csr_structures()

        self._id = uuid.uuid4().hex  # Reset graph ID since structure changed
        logger.info(f"Set {len(turn_restrictions)} turn restrictions, {allow_path_uturns=}")

    def clear_turn_restrictions(self) -> None:
        """Clears all turn restrictions from the graph."""
        self._turn_restrictions = None
        self._allow_path_uturns = False
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
        Generates prohibited turns between centroid connectors meeting at the same node.
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
            if node < 0 or node + 1 >= fs.shape[0]:
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
        Builds CSR structures for turn transitions in the compact graph.

        For arc-based Dijkstra, we need to map from each incoming arc to its
        possible outgoing arcs with associated turn penalties.
        """
        if self._turn_restrictions is not None and len(self._turn_restrictions) > 0:
            tr = self._turn_restrictions
            tr_from_node = tr["from_node"].to_numpy(np.int64, copy=False)
            tr_via_node = tr["via_node"].to_numpy(np.int64, copy=False)
            tr_to_node = tr["to_node"].to_numpy(np.int64, copy=False)
            tr_penalty = pd.to_numeric(tr["penalty"], errors="coerce").to_numpy(np.float64, copy=False)

            max_idx = self.nodes_to_indices.shape[0] - 1
            valid_nodes = (
                (tr_from_node >= 0)
                & (tr_via_node >= 0)
                & (tr_to_node >= 0)
                & (tr_from_node <= max_idx)
                & (tr_via_node <= max_idx)
                & (tr_to_node <= max_idx)
            )

            tr_from_node_full = np.where(valid_nodes, self.nodes_to_indices[tr_from_node], -1)
            tr_via_node_full = np.where(valid_nodes, self.nodes_to_indices[tr_via_node], -1)
            tr_to_node_full = np.where(valid_nodes, self.nodes_to_indices[tr_to_node], -1)
        else:
            tr_from_node = np.empty(0, dtype=np.int64)
            tr_via_node = np.empty(0, dtype=np.int64)
            tr_to_node = np.empty(0, dtype=np.int64)
            tr_penalty = np.empty(0, dtype=np.float64)
            tr_from_node_full = np.empty(0, dtype=np.int64)
            tr_via_node_full = np.empty(0, dtype=np.int64)
            tr_to_node_full = np.empty(0, dtype=np.int64)

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
                tr_penalty,
            )
        else:
            full_from_arcs = np.empty(0, dtype=np.int64)
            full_to_arcs = np.empty(0, dtype=np.int64)
            full_penalties = np.empty(0, dtype=np.float64)

        # 2) Auto-generate bans between centroid connectors when centroid flows are blocked.
        if self._has_multi_connector_centroid():
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

        if self.compact_graph.empty:
            self.compact_turn_fs = np.array([], dtype=self.default_types("int"))
            self.compact_turn_to_arcs = np.array([], dtype=self.default_types("int"))
            self.compact_turn_penalties = np.array([], dtype=self.default_types("float"))
            self._compact_turn_penalties_master = np.array(self.compact_turn_penalties, copy=True)
            self._has_turn_restrictions = self.turn_penalties.size > 0
            return

        num_compact_links = self.compact_num_links
        compact_ids = self.compact_graph["id"].to_numpy(np.int64, copy=False)
        compact_a_by_id = np.empty(num_compact_links, dtype=np.int64)
        compact_b_by_id = np.empty(num_compact_links, dtype=np.int64)
        compact_a_by_id[compact_ids] = self.compact_graph["a_node"].to_numpy(np.int64, copy=False)
        compact_b_by_id[compact_ids] = self.compact_graph["b_node"].to_numpy(np.int64, copy=False)

        # 4) Repeat mapping for compact graph IDs used by assignment/skimming.
        if tr_from_node.size > 0 and self.compact_nodes_to_indices.shape[0] > 0:
            max_compact_idx = self.compact_nodes_to_indices.shape[0] - 1
            valid_compact_nodes = (
                (tr_from_node >= 0)
                & (tr_via_node >= 0)
                & (tr_to_node >= 0)
                & (tr_from_node <= max_compact_idx)
                & (tr_via_node <= max_compact_idx)
                & (tr_to_node <= max_compact_idx)
            )
            tr_from_node_compact = np.where(valid_compact_nodes, self.compact_nodes_to_indices[tr_from_node], -1)
            tr_via_node_compact = np.where(valid_compact_nodes, self.compact_nodes_to_indices[tr_via_node], -1)
            tr_to_node_compact = np.where(valid_compact_nodes, self.compact_nodes_to_indices[tr_to_node], -1)

            compact_from_arcs, compact_to_arcs, compact_penalties = self._map_node_turn_restrictions_to_arcs(
                self.compact_fs,
                compact_a_by_id,
                compact_b_by_id,
                tr_from_node_compact,
                tr_via_node_compact,
                tr_to_node_compact,
                tr_penalty,
            )
        else:
            compact_from_arcs = np.empty(0, dtype=np.int64)
            compact_to_arcs = np.empty(0, dtype=np.int64)
            compact_penalties = np.empty(0, dtype=np.float64)

        if self._has_multi_connector_centroid():
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

        self._has_turn_restrictions = self.turn_penalties.size > 0 or self.compact_turn_penalties.size > 0

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
        incoming_by_leg: dict[tuple[int, int], list[int]] = {}

        num_arcs = a_by_arc.shape[0]
        for arc_id in range(num_arcs):
            a_node = int(a_by_arc[arc_id])
            b_node = int(b_by_arc[arc_id])
            incoming_by_leg.setdefault((a_node, b_node), []).append(arc_id)

        out_from: list[int] = []
        out_to: list[int] = []
        out_penalty: list[float] = []

        for i in range(from_nodes.shape[0]):
            fn = int(from_nodes[i])
            vn = int(via_nodes[i])
            tn = int(to_nodes[i])
            incoming = incoming_by_leg.get((fn, vn), [])
            outgoing = incoming_by_leg.get((vn, tn), [])
            if not incoming or not outgoing:
                continue

            for from_arc in incoming:
                head = int(b_by_arc[from_arc])
                if head < 0 or head + 1 >= fs.shape[0]:
                    continue
                out_set = set(range(int(fs[head]), int(fs[head + 1])))
                for to_arc in outgoing:
                    if to_arc in out_set:
                        out_from.append(from_arc)
                        out_to.append(to_arc)
                        out_penalty.append(float(penalties[i]))

        if len(out_from) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        return (
            np.asarray(out_from, dtype=np.int64),
            np.asarray(out_to, dtype=np.int64),
            np.asarray(out_penalty, dtype=np.float64),
        )

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

        for f_arc, t_arc, penalty in zip(restricted_from_arcs, restricted_to_arcs, restricted_penalties, strict=True):
            f_arc = int(f_arc)
            t_arc = int(t_arc)
            if f_arc < 0 or t_arc < 0 or f_arc >= num_arcs or t_arc >= num_arcs:
                continue

            key = (f_arc, t_arc)
            if penalty < 0:
                raise ValueError("Negative turn penalties are not allowed. Use None/NaN/+inf for a prohibition")
            # Convention: NaN/non-finite means prohibited turn.
            if not np.isfinite(penalty):
                prohibited_keys.add(key)
                key_to_penalty.pop(key, None)
            elif key not in prohibited_keys:
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
            self.description = mygraph["description"]
            self.num_links = mygraph["num_links"]
            self.num_nodes = mygraph["num_nodes"]
            self.network = mygraph["network"]
            self.graph = mygraph["graph"]
            self.turn_penalty_dimension = mygraph.get("turn_penalty_dimension", "time")
            self.turn_skim_fields = mygraph.get("turn_skim_fields", [])

            self.all_nodes = mygraph["all_nodes"]
            self.nodes_to_indices = mygraph["nodes_to_indices"]
            self.num_nodes = mygraph["num_nodes"]
            self.fs = mygraph["fs"]
            self.cost = mygraph["cost"]
            self.cost_field = mygraph["cost_field"]
            self.skims = mygraph["skims"]
            self.skim_fields = mygraph["skim_fields"]
            self.block_centroid_flows = mygraph["block_centroid_flows"]
            self.centroids = mygraph["centroids"]
            self._id = mygraph["graph_id"]
            self.mode = mygraph["mode"]

            # Load turn restrictions data (if present in saved file)
            self._turn_restrictions = mygraph.get("turn_restrictions", None)
            # Support loading graphs saved with old field name "allow_uturns"
            self._allow_uturns_everywhere = mygraph.get("allow_uturns_everywhere", mygraph.get("allow_uturns", False))
            self._allow_path_uturns = mygraph.get("allow_path_uturns", False)
            self._has_turn_restrictions = mygraph.get("has_turn_restrictions", False)
            self.turn_fs = mygraph.get("turn_fs", np.array([]))
            self.turn_to_arcs = mygraph.get("turn_to_arcs", np.array([]))
            self.turn_penalties = mygraph.get("turn_penalties", np.array([]))
            self.compact_turn_fs = mygraph.get("compact_turn_fs", np.array([]))
            self.compact_turn_to_arcs = mygraph.get("compact_turn_to_arcs", np.array([]))
            self.compact_turn_penalties = mygraph.get("compact_turn_penalties", np.array([]))
        self._backup_penalties()

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
