import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from aequilibrae.paths.AoN import sum_axis1, assign_link_loads

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.paths.graph import Graph, TransitGraph, GraphBase, _get_graph_to_network_mapping

"""
TO-DO:
1. Make the writing to SQL faster by disabling all checks before the actual writing
"""


class AssignmentResultsBase(ABC):
    """Assignment results base class for traffic and transit assignments."""

    def __init__(self):
        self.link_loads = np.array([])  # The actual results for assignment
        self.no_path = None  # The list os paths
        self.num_skims = 0  # number of skims that will be computed. Depends on the setting of the graph provided
        p = Parameters().parameters["system"]["cpus"]
        if not isinstance(p, int):
            p = 0
        self.set_cores(p)

        self.nodes = -1
        self.zones = -1
        self.links = -1

        self.lids = None

    @abstractmethod
    def prepare(self, graph: GraphBase, matrix: AequilibraeMatrix) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def set_cores(self, cores: int) -> None:
        """
        Sets number of cores (threads) to be used in computation

        Value of zero sets number of threads to all available in the system, while negative values indicate the number
        of threads to be left out of the computational effort.

        Resulting number of cores will be adjusted to a minimum of zero or the maximum available in the system if the
        inputs result in values outside those limits

        :Arguments:
            **cores** (:obj:`int`): Number of cores to be used in computation
        """

        if not isinstance(cores, int):
            raise ValueError("Number of cores needs to be an integer")

        if cores < 0:
            self.cores = max(1, mp.cpu_count() + cores)
        elif cores == 0:
            self.cores = mp.cpu_count()
        elif cores > 0:
            cores = min(mp.cpu_count(), cores)
            if self.cores != cores:
                self.cores = cores
        if self.link_loads.shape[0]:
            self.__redim()


class AssignmentResults(AssignmentResultsBase):
    """
    Assignment result holder for a single :obj:`TrafficClass` with multiple user classes
    """

    def __init__(self):
        super().__init__()
        self.compact_link_loads = np.array([])  # Results for assignment on simplified graph
        self.compact_total_link_loads = np.array([])  # Results for all user classes summed on simplified graph
        self.crosswalk = np.array([])  # crosswalk between compact graph link IDs and actual link IDs
        self.skims = AequilibraeMatrix()  # The array of skims
        self.total_link_loads = np.array([])  # The result of the assignment for all user classes summed

        self.compact_links = -1
        self.compact_nodes = -1

        self.direcs = None

        self.classes = {"number": 1, "names": ["flow"]}

        self._selected_links = {}
        self.select_link_od = None
        self.select_link_loading = {}

        self._graph_id = None
        self.__float_type = None
        self.__integer_type = None

        # save path files. Need extra metadata for file paths
        self.save_path_file = False
        self.path_file_dir = None
        self.write_feather = True  # we use feather as default, parquet is slower but with better compression

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph: Graph, matrix: AequilibraeMatrix) -> None:
        """
        Prepares the object with dimensions corresponding to the assignment matrix and graph objects

        :Arguments:
            **graph** (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)

            **matrix** (:obj:`AequilibraeMatrix`): Matrix properly set for computation with
            ``matrix.computational_view(:obj:`list`)``
        """

        self.__float_type = graph.default_types("float")
        self.__integer_type = graph.default_types("int")

        if matrix.view_names is None:
            raise ValueError("Please set the matrix_procedures computational view")
        self.classes["number"] = 1
        if len(matrix.matrix_view.shape) > 2:
            self.classes["number"] = matrix.matrix_view.shape[2]
        self.classes["names"] = matrix.view_names

        if graph is None:
            raise ValueError("Please provide a graph")
        self.compact_nodes = graph.compact_num_nodes
        self.compact_links = graph.compact_num_links

        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.num_skims = len(graph.skim_fields)
        self.skim_names = list(graph.skim_fields)
        self.lids = graph.graph.link_id.values
        self.direcs = graph.graph.direction.values
        self.crosswalk = np.zeros(graph.graph.shape[0], self.__integer_type)
        self.crosswalk[graph.graph.__supernet_id__.values] = graph.graph.__compressed_id__.values
        self._graph_ids = graph.graph.__supernet_id__.values
        self._graph_compressed_ids = graph.graph.__compressed_id__.values
        self.__redim()
        self._graph_id = graph._id

        if self._selected_links:
            self.select_link_od = AequilibraeMatrix()
            self.select_link_od.create_empty(
                memory_only=True,
                zones=matrix.zones,
                matrix_names=list(self._selected_links.keys()),
                index_names=matrix.index_names,
            )

            self.select_link_loading = {}
            # Combine each set of selected links into one large matrix that can be parsed into Cython
            # Each row corresponds a link set, and the equivalent rows in temp_sl_od_matrix and temp_sl_link_loading
            # Correspond to that set
            self.select_links = np.full(
                (len(self._selected_links), max([len(x) for x in self._selected_links.values()])),
                -1,
                dtype=graph.default_types("int"),
            )

            sl_idx = {}
            for i, (name, arr) in enumerate(self._selected_links.items()):
                sl_idx[name] = i
                # Filling select_links array with linksets. Note the default value is -1, which is used as a placeholder
                # It also denotes when the given row has no more selected links, since Cython cannot handle
                # Multidimensional arrays where each row has different lengths
                self.select_links[i][: len(arr)] = arr
                # Correctly sets the dimensions for the final output matrices
                self.select_link_od.matrix[name] = np.zeros(
                    (graph.num_zones, graph.num_zones, self.classes["number"]),
                    dtype=graph.default_types("float"),
                )
                self.select_link_loading[name] = np.zeros(
                    (graph.compact_num_links, self.classes["number"]),
                    dtype=graph.default_types("float"),
                )

            # Overwrites previous arrays on assignment results level with the index to access that array in Cython
            self._selected_links = sl_idx

    def reset(self) -> None:
        """
        Resets object to prepared and pre-computation state
        """
        if self.num_skims > 0:
            self.skims.matrices.fill(0)
        if self.link_loads is not None:
            self.no_path.fill(0)
            self.link_loads.fill(0)
            self.total_link_loads.fill(0)
            self.compact_link_loads.fill(0)
            self.compact_total_link_loads.fill(0)
        else:
            raise ValueError("Exception: Assignment results object was not yet prepared/initialized")

    def __redim(self):
        self.compact_link_loads = np.zeros((self.compact_links + 1, self.classes["number"]), self.__float_type)
        self.compact_total_link_loads = np.zeros(self.compact_links, self.__float_type)

        self.link_loads = np.zeros((self.links, self.classes["number"]), self.__float_type)
        self.total_link_loads = np.zeros(self.links, self.__float_type)
        self.no_path = np.zeros((self.zones, self.zones), dtype=self.__integer_type)

        if self.num_skims > 0:
            self.skims = AequilibraeMatrix()

            self.skims.create_empty(file_name=self.skims.random_name(), zones=self.zones, matrix_names=self.skim_names)
            self.skims.index[:] = self.centroids[:]
            self.skims.computational_view()
            if len(self.skims.matrix_view.shape[:]) == 2:
                self.skims.matrix_view = self.skims.matrix_view.reshape((self.zones, self.zones, 1))
        else:
            self.skims = AequilibraeMatrix()
            self.skims.matrix_view = np.array((1, 1, 1))

        self.reset()

    def total_flows(self) -> None:
        """
        Totals all link flows for this class into a single link load

        Results are placed into *total_link_loads* class member
        """
        sum_axis1(self.total_link_loads, self.link_loads, self.cores)

    def get_graph_to_network_mapping(self):
        return _get_graph_to_network_mapping(self.lids, self.direcs)

    def get_load_results(self) -> pd.DataFrame:
        """
        Translates the assignment results from the graph format into the network format

        :Returns:
            **dataset** (:obj:`pd.DataFrame`): Pandas DataFrame data with the traffic class assignment results
        """

        # Get a mapping from the compressed graph to/from the network graph
        m = self.get_graph_to_network_mapping()

        recs = np.unique(self.lids).shape[0]

        # Link flows
        link_flows = self.link_loads[self._graph_ids, :]
        aux = {}
        for i, n in enumerate(self.classes["names"]):
            # Directional Flows
            aux[n + "_ab"] = np.zeros(recs, self.__float_type)
            aux[n + "_ab"][m.network_ab_idx] = np.nan_to_num(link_flows[m.graph_ab_idx, i])

            aux[n + "_ba"] = np.zeros(recs, self.__float_type)
            aux[n + "_ba"][m.network_ba_idx] = np.nan_to_num(link_flows[m.graph_ba_idx, i])

            # Tot Flow
            aux[n + "_tot"] = np.nan_to_num(aux[n + "_ab"]) + np.nan_to_num(aux[n + "_ba"])

        return pd.DataFrame(aux, index=np.unique(self.lids))

    def get_sl_results(self) -> pd.DataFrame:
        # Set up the name for each column. Each set of select links has a column for ab, ba, total flows
        # for each subclass contained in the TrafficClass
        fields = [
            e
            for name in self._selected_links.keys()
            for n in self.classes["names"]
            for e in [f"{name}_{n}_ab", f"{name}_{n}_ba", f"{name}_{n}_tot"]
        ]

        res = pd.DataFrame([], columns=fields, index=np.unique(self.lids))

        m = self.get_graph_to_network_mapping()
        for name in self._selected_links.keys():
            # Link flows initialised
            link_flows = np.full((self.links, self.classes["number"]), np.nan)
            # maps link flows from the compressed graph to the uncompressed graph
            assign_link_loads(link_flows, self.select_link_loading[name], self._graph_compressed_ids, self.cores)
            for i, n in enumerate(self.classes["names"]):
                # Directional Flows
                res[f"{name}_{n}_ab"].values[m.network_ab_idx] = link_flows[m.graph_ab_idx, i]
                res[f"{name}_{n}_ba"].values[m.network_ba_idx] = link_flows[m.graph_ba_idx, i]

                # Tot Flow
                res[f"{name}_{n}_tot"] = np.nansum(res[[f"{name}_{n}_ab", f"{name}_{n}_ba"]].to_numpy(), axis=1)

        return res


class TransitAssignmentResults(AssignmentResultsBase):
    """
    Assignment result holder for a single :obj:`Transit`
    """

    def __init__(self):
        super().__init__()

        self.link_loads = np.array([])

    def prepare(self, graph: TransitGraph, matrix: AequilibraeMatrix) -> None:
        """
        Prepares the object with dimensions corresponding to the assignment matrix and graph objects

        :Arguments:
            **graph** (:obj:`TransitGraph`): Needs to have been set with number of centroids

            **matrix** (:obj:`AequilibraeMatrix`): Matrix properly set for computation with
            ``matrix.computational_view(:obj:`list`)``
        """
        self.reset()
        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.lids = graph.graph.link_id.values

    def reset(self) -> None:
        """
        Resets object to prepared and pre-computation state
        """

        # Since all memory for the assignment is managed by the HyperpathGenerating
        # object we don't need to do much here
        self.link_loads.fill(0)

    def get_load_results(self) -> pd.DataFrame:
        """
        Translates the assignment results from the graph format into the network format

        :Returns:
            **dataset** (:obj:`pd.DataFrame`): DataFrame data with the transit class assignment results
        """
        if not self.link_loads.shape[0]:
            raise ValueError("Transit assignment has not been executed yet")

        return pd.DataFrame({"volume": self.link_loads}, index=self.lids)
