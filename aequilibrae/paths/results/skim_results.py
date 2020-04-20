import numpy as np
import multiprocessing as mp
from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming


class SkimResults:
    """
    Network skimming result holder

    ::

          from aequilibrae.project import Project
          from aequilibrae.paths.results import SkimResults

          proj = Project()
          proj.load('path/to/project.sqlite')
          proj.network.build_graphs()
          # Mode c is car in this project
          car_graph = proj.network.graphs['c']

          # minimize travel time
          car_graph.set_graph('free_flow_travel_time')

          # Skims travel time and distance
          car_graph.set_skimming(['free_flow_travel_time', 'distance'])

          res = SkimResults()
          res.prepare(car_graph)
          res.compute_skims()

          res.skims.export('path/to/matrix.aem')
    """

    def __init__(self):
        self.skims = None
        self.path = None
        self.path_nodes = None
        self.milepost = None
        self.cores = mp.cpu_count()

        self.links = -1
        self.nodes = -1
        self.zones = -1
        self.num_skims = -1
        self.__graph_id__ = None
        self.graph: Graph = None

    def prepare(self, graph):
        """
        Prepares the object with dimensions corresponding to the graph objects

        Args:
            *graph* (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)
        """

        if not graph.cost_field:
            raise Exception('Cost field needs to be set for computation. use graph.set_graph("your_cost_field")')

        self.nodes = graph.num_nodes + 1
        self.zones = graph.num_zones
        self.links = graph.num_links + 1
        self.num_skims = len(graph.skim_fields)

        self.skims = AequilibraeMatrix()
        self.skims.create_empty(
            file_name=AequilibraeMatrix().random_name(),
            zones=self.zones,
            matrix_names=graph.skim_fields,
        )
        self.skims.index[:] = graph.centroids[:]
        self.skims.computational_view(core_list=self.skims.names)
        self.skims.matrix_view = self.skims.matrix_view.reshape(
            self.zones, self.zones, self.num_skims
        )
        self.__graph_id__ = graph.__id__
        self.graph = graph

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
        else:
            raise ValueError("Number of cores needs to be an integer")

    def reset(self) -> None:
        """
        Resets object to prepared and pre-computation state
        """
        if self.skims is not None:
            self.skims.fill(np.inf)
            self.path = None
            self.path_nodes = None
            self.milepost = None

        else:
            raise ValueError(
                "Exception: Path results object was not yet prepared/initialized"
            )

    def compute_skims(self) -> None:
        """Computes the skims as set"""

        ns = NetworkSkimming(self.graph, self)
        ns.execute()
