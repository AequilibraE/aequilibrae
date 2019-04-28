# cimport numpy as np
# cimport cython
# include 'parameters.pxi'
import numpy as np
import multiprocessing as mp
from ...matrix import AequilibraeMatrix


class SkimResults:
    def __init__(self):
        """
        @type graph: Set of numpy arrays to store Computation results
        """
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

    def prepare(self, graph):

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

    def set_cores(self, cores):
        if isinstance(cores, int):
            if cores > 0:
                if self.cores != cores:
                    self.cores = cores
            else:
                raise ValueError("Number of cores needs to be equal or bigger than one")
        else:
            raise ValueError("Number of cores needs to be an integer")

    def reset(self):
        if self.skims is not None:
            self.skims.fill(np.inf)
            self.path = None
            self.path_nodes = None
            self.milepost = None

        else:
            raise ValueError(
                "Exception: Path results object was not yet prepared/initialized"
            )
