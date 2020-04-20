import multiprocessing as mp
import numpy as np
import warnings
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
from aequilibrae.paths.graph import Graph
from aequilibrae.parameters import Parameters
from aequilibrae import logger

try:
    from aequilibrae.paths.AoN import sum_axis1
except ImportError as ie:
    logger.warning(f'Could not import procedures from the binary. {ie.args}')

"""
TO-DO:
1. Create a file type for memory-mapped path files
   Same idea of the AequilibraEData container, but using the format.memmap from NumPy
2. Make the writing to SQL faster by disabling all checks before the actual writing
"""


class AssignmentResults:
    """
    Assignment result holder for a single :obj:`TrafficClass` with multiple user classes
    """

    def __init__(self):
        self.link_loads = None  # type: np.array  # The actual results for assignment
        self.total_link_loads = None  # type: np.array  # The result of the assignment for all user classes summed
        self.skims = None  # The array of skims
        self.no_path = None  # The list os paths
        self.num_skims = None  # number of skims that will be computed. Depends on the setting of the graph provided
        p = Parameters().parameters['system']['cpus']
        if not isinstance(p, int):
            p = 0
        self.set_cores(p)

        self.classes = {"number": 1, "names": ["flow"]}

        self.critical_links = {"save": False, "queries": {}, "results": False}  # Queries are a dictionary

        self.link_extraction = {"save": False, "queries": {}, "output": None}  # Queries are a dictionary

        self.path_file = {"save": False, "results": None}

        self.nodes = -1
        self.zones = -1
        self.links = -1
        self.__graph_id__ = None
        self.__float_type = None
        self.__integer_type = None

        self.lids = None
        self.direcs = None

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph: Graph, matrix: AequilibraeMatrix) -> None:
        """
        Prepares the object with dimensions corresponding to the assignment matrix and graph objects

        Args:
            *graph* (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)

            *matrix* (:obj:`AequilibraeMatrix`): Matrix properly set for computation with
             matrix.computational_view(:obj:`list`)
        """

        self.__float_type = graph.default_types("float")
        self.__integer_type = graph.default_types("int")

        if matrix.view_names is None:
            raise ("Please set the matrix_procedures computational view")
        else:
            self.classes["number"] = 1
            if len(matrix.matrix_view.shape) > 2:
                self.classes["number"] = matrix.matrix_view.shape[2]
            self.classes["names"] = matrix.view_names

        if graph is None:
            raise ("Please provide a graph")
        else:
            if not graph.cost_field:
                raise Exception('Cost field needs to be set for computation. use graph.set_graph("your_cost_field")')

            self.nodes = graph.num_nodes
            self.zones = graph.num_zones
            self.centroids = graph.centroids
            self.links = graph.num_links
            self.num_skims = len(graph.skim_fields)
            self.skim_names = [x for x in graph.skim_fields]
            self.lids = graph.graph["link_id"]
            self.direcs = graph.graph["direction"]
            self.__redim()
            self.__graph_id__ = graph.__id__

            # TODO: Enable these methods when the work for select link analysis and saving path files is completed
            self.__setSavePathFile(False)
            self.__setCriticalLinks(False)

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
        else:
            raise ValueError("Exception: Assignment results object was not yet prepared/initialized")

    def __redim(self):
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
        """ Totals all link flows for this class into a single link load

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

        Args:
            *cores* (:obj:`int`): Number of cores to be used in computation
        """

        if isinstance(cores, int):
            if cores < 0:
                self.cores = max(1, mp.cpu_count() + cores)
            if cores == 0:
                self.cores = mp.cpu_count()
            elif cores > 0:
                cores = max(mp.cpu_count(), cores)
                if self.cores != cores:
                    self.cores = cores
                    if self.link_loads is not None:
                        self.__redim()
        else:
            raise ValueError("Number of cores needs to be an integer")

    def __setCriticalLinks(self, save=False, queries={}, crit_res_result=None):
        a = AequilibraeMatrix()
        if save:
            if crit_res_result is None:
                warnings.warn("Critical Link analysis not set properly. Need to specify output file too")
            else:
                if crit_res_result[-3:].lower() != "aem":
                    crit_res_result += ".aes"

                if self.nodes > 0 and self.zones > 0:
                    if ["elements", "labels", "type"] in queries.keys():
                        if len(queries["labels"]) == len(queries["elements"]) == len(queries["type"]):
                            a.create_empty(file_name=crit_res_result, zones=self.zones, matrix_names=queries["labels"])
                        else:
                            raise ValueError(
                                "Queries are inconsistent. 'Labels', 'elements' and 'type' need to have same dimensions"
                            )
                    else:
                        raise ValueError(
                            "Queries are inconsistent. It needs to contain the following elements: 'Labels', 'elements' and 'type'"
                        )
        else:
            a.create_empty(file_name=a.random_name(), zones=self.zones, matrix_names=["empty", "nothing"])

        a.computational_view()
        if len(a.matrix_view.shape[:]) == 2:
            a.matrix_view = a.matrix_view.reshape((self.zones, self.zones, 1))
        self.critical_links = {"save": save, "queries": queries, "results": a}

    def __setSavePathFile(self, save=False, path_result=None):
        # Fields: Origin, Node, Predecessor
        # Number of records: Origins * Nodes
        a = AequilibraeData()
        d1 = max(1, self.zones)
        d2 = 1
        memory_mode = True

        if save:
            if path_result is None:
                warnings.warn("Path file not set properly. Need to specify output file too")
            else:
                # This is the only place where we keep 32bits, as going 64 bits would explode the file size
                if self.nodes > 0 and self.zones > 0:
                    d1 = self.zones
                    d2 = self.nodes
                    memory_mode = False

        a.create_empty(
            file_path=path_result,
            entries=d1 * d2,
            field_names=["origin", "node", "predecessor", "connector"],
            data_types=[np.uint32, np.uint32, np.uint32, np.uint32],
            memory_mode=memory_mode,
        )

        self.path_file = {"save": save, "results": a}

    def get_load_results(self) -> AequilibraeData:
        """
        Translates the assignment results from the graph format into the network format

        Returns:
            dataset (:obj:`AequilibraeData`): AequilibraE data with the traffic class assignment results
        """
        fields = ['link_id']
        for n in self.classes['names']:
            fields.extend([f'{n}_ab', f'{n}_ba', f'{n}_tot'])
        types = [np.float64] * len(fields)

        entries = int(np.unique(self.lids).shape[0])
        res = AequilibraeData()
        res.create_empty(memory_mode=True, entries=entries, field_names=fields, data_types=types)
        res.data.fill(np.nan)
        res.index[:] = np.unique(self.lids)[:]
        res.link_id[:] = res.index[:]

        indexing = np.zeros(int(self.lids.max()) + 1, np.uint64)
        indexing[res.index[:]] = np.arange(entries)

        # Indices of links BA and AB
        ABs = self.direcs > 0
        BAs = self.direcs < 0
        ab_ids = indexing[self.lids[ABs]]
        ba_ids = indexing[self.lids[BAs]]

        # Link flows
        link_flows = self.link_loads[:, :]
        for i, n in enumerate(self.classes["names"]):
            # AB Flows
            res.data[n + "_ab"][ab_ids] = np.nan_to_num(link_flows[ABs, i])
            # BA Flows
            res.data[n + "_ba"][ba_ids] = np.nan_to_num(link_flows[BAs, i])

            # Tot Flow
            res.data[n + "_tot"] = np.nan_to_num(res.data[n + "_ab"]) + np.nan_to_num(res.data[n + "_ba"])
        return res

    def save_to_disk(self, file_name=None, output="loads") -> None:
        """ Function to write to disk all outputs computed during assignment

        Args:
            *file_name* (:obj:`str`): Name of the file, with extension. Valid extensions are: ['aed', 'csv', 'sqlite']
            *output* (:obj:`str`, optional): Type of output ('loads', 'path_file'). Defaults to 'loads'
        """

        if output == "loads":
            res = self.get_load_results()
            res.export(file_name)

        # TODO: Re-factor the exporting of the path file within the AequilibraeData format
        elif output == "path_file":
            pass
