import dataclasses
import multiprocessing as mp

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
from aequilibrae.paths.graph import Graph
from aequilibrae.parameters import Parameters
from aequilibrae import global_logger
from pathlib import Path

try:
    from aequilibrae.paths.AoN import sum_axis1, assign_link_loads
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

"""
TO-DO:
1. Create a file type for memory-mapped path files
   Same idea of the AequilibraEData container, but using the format.memmap from NumPy
2. Make the writing to SQL faster by disabling all checks before the actual writing
"""


@dataclasses.dataclass
class NetworkGraphIndices:
    network_ab_idx: np.array
    network_ba_idx: np.array
    graph_ab_idx: np.array
    graph_ba_idx: np.array


class AssignmentResults:
    """
    Assignment result holder for a single :obj:`TrafficClass` with multiple user classes
    """

    def __init__(self):
        self.compact_link_loads = np.array([])  # Results for assignment on simplified graph
        self.compact_total_link_loads = np.array([])  # Results for all user classes summed on simplified graph
        self.link_loads = np.array([])  # The actual results for assignment
        self.total_link_loads = np.array([])  # The result of the assignment for all user classes summed
        self.crosswalk = np.array([])  # crosswalk between compact graph link IDs and actual link IDs
        self.skims = AequilibraeMatrix()  # The array of skims
        self.no_path = None  # The list os paths
        self.num_skims = 0  # number of skims that will be computed. Depends on the setting of the graph provided
        p = Parameters().parameters["system"]["cpus"]
        if not isinstance(p, int):
            p = 0
        self.set_cores(p)

        self.classes = {"number": 1, "names": ["flow"]}

        self._selected_links = {}
        self.select_link_od = None
        self.select_link_loading = {}

        self.nodes = -1
        self.zones = -1
        self.links = -1
        self.compact_links = -1
        self.compact_nodes = -1
        self.__graph_id__ = None
        self.__float_type = None
        self.__integer_type = None

        self.lids = None
        self.direcs = None

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
             matrix.computational_view(:obj:`list`)
        """

        self.__float_type = graph.default_types("float")
        self.__integer_type = graph.default_types("int")

        if matrix.view_names is None:
            raise ("Please set the matrix_procedures computational view")
        self.classes["number"] = 1
        if len(matrix.matrix_view.shape) > 2:
            self.classes["number"] = matrix.matrix_view.shape[2]
        self.classes["names"] = matrix.view_names

        if graph is None:
            raise ("Please provide a graph")
        self.compact_nodes = graph.compact_num_nodes
        self.compact_links = graph.compact_num_links

        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.num_skims = len(graph.skim_fields)
        self.skim_names = [x for x in graph.skim_fields]
        self.lids = graph.graph.link_id.values
        self.direcs = graph.graph.direction.values
        self.crosswalk = np.zeros(graph.graph.shape[0], self.__integer_type)
        self.crosswalk[graph.graph.__supernet_id__.values] = graph.graph.__compressed_id__.values
        self.__graph_ids = graph.graph.__supernet_id__.values
        self.__graph_compressed_ids = graph.graph.__compressed_id__.values
        self.__redim()
        self.__graph_id__ = graph.__id__

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
            for i, val in enumerate(self._selected_links.items()):
                name, arr = val
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
        """Totals all link flows for this class into a single link load

        Results are placed into *total_link_loads* class member
        """
        sum_axis1(self.total_link_loads, self.link_loads, self.cores)

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

    def get_graph_to_network_mapping(self):
        num_uncompressed_links = int(np.unique(self.lids).shape[0])
        indexing = np.zeros(int(self.lids.max()) + 1, np.uint64)
        indexing[np.unique(self.lids)[:]] = np.arange(num_uncompressed_links)

        graph_ab_idx = self.direcs > 0
        graph_ba_idx = self.direcs < 0
        network_ab_idx = indexing[self.lids[graph_ab_idx]]
        network_ba_idx = indexing[self.lids[graph_ba_idx]]
        return NetworkGraphIndices(network_ab_idx, network_ba_idx, graph_ab_idx, graph_ba_idx)

    def get_load_results(self) -> AequilibraeData:
        """
        Translates the assignment results from the graph format into the network format

        :Returns:
            **dataset** (:obj:`AequilibraeData`): AequilibraE data with the traffic class assignment results
        """
        fields = [e for n in self.classes["names"] for e in [f"{n}_ab", f"{n}_ba", f"{n}_tot"]]
        types = [np.float64] * len(fields)

        # Create a data store with a row for each uncompressed link
        res = AequilibraeData.empty(
            memory_mode=True,
            entries=int(np.unique(self.lids).shape[0]),
            field_names=fields,
            data_types=types,
            fill=np.nan,
            index=np.unique(self.lids),
        )

        # Get a mapping from the compressed graph to/from the network graph
        m = self.get_graph_to_network_mapping()

        # Link flows
        link_flows = self.link_loads[self.__graph_ids, :]
        for i, n in enumerate(self.classes["names"]):
            # Directional Flows
            res.data[n + "_ab"][m.network_ab_idx] = np.nan_to_num(link_flows[m.graph_ab_idx, i])
            res.data[n + "_ba"][m.network_ba_idx] = np.nan_to_num(link_flows[m.graph_ba_idx, i])

            # Tot Flow
            res.data[n + "_tot"] = np.nan_to_num(res.data[n + "_ab"]) + np.nan_to_num(res.data[n + "_ba"])

        return res

    def get_sl_results(self) -> AequilibraeData:
        # Set up the name for each column. Each set of select links has a column for ab, ba, total flows
        # for each subclass contained in the TrafficClass
        fields = [
            e
            for name in self._selected_links.keys()
            for n in self.classes["names"]
            for e in [f"{name}_{n}_ab", f"{name}_{n}_ba", f"{name}_{n}_tot"]
        ]
        types = [np.float64] * len(fields)
        # Create a data store with a row for each uncompressed link, columns for each set of select links
        res = AequilibraeData.empty(
            memory_mode=True,
            entries=int(np.unique(self.lids).shape[0]),
            field_names=fields,
            data_types=types,
            fill=np.nan,
            index=np.unique(self.lids),
        )
        m = self.get_graph_to_network_mapping()
        for name in self._selected_links.keys():
            # Link flows initialised
            link_flows = np.full((self.links, self.classes["number"]), np.nan)
            # maps link flows from the compressed graph to the uncompressed graph
            assign_link_loads(link_flows, self.select_link_loading[name], self.__graph_compressed_ids, self.cores)
            for i, n in enumerate(self.classes["names"]):
                # Directional Flows
                res.data[name + "_" + n + "_ab"][m.network_ab_idx] = np.nan_to_num(link_flows[m.graph_ab_idx, i])
                res.data[name + "_" + n + "_ba"][m.network_ba_idx] = np.nan_to_num(link_flows[m.graph_ba_idx, i])

                # Tot Flow
                res.data[name + "_" + n + "_tot"] = np.nan_to_num(res.data[name + "_" + n + "_ab"]) + np.nan_to_num(
                    res.data[name + "_" + n + "_ba"]
                )

        return res

    def save_to_disk(self, file_name=None, output="loads") -> None:
        """
        Function to write to disk all outputs computed during assignment.

        .. deprecated:: 0.7.0

        :Arguments:
            **file_name** (:obj:`str`): Name of the file, with extension. Valid extensions are: ['aed', 'csv', 'sqlite']
            **output** (:obj:`str`, optional): Type of output ('loads', 'path_file'). Defaults to 'loads'
        """

        if output == "loads":
            res = self.get_load_results()
            res.export(file_name)

        # TODO: Re-factor the exporting of the path file within the AequilibraeData format
        elif output == "path_file":
            raise NotImplementedError
