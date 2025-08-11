import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.paths.graph import GraphBase

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
