from typing import Union
import numpy as np
from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.results import AssignmentResults


class TrafficClass():
    """Traffic class for equilibrium traffic assignment

    ::

        from aequilibrae.paths import TrafficClass

        tc = TrafficClass(graph, demand_matrix)
        tc.set_pce(1.3)
    """
    def __init__(self, graph: Graph, matrix: AequilibraeMatrix) -> None:
        """
        Instantiates the class

         Args:
            graph (:obj:`Graph`): Class/mode-specific graph

            matrix (:obj:`AequilibraeMatrix`): Class/mode-specific matrix. Supports multiple user classes
        """
        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

        self.graph = graph
        self.matrix = matrix
        self.pce = 1
        self.mode = graph.mode
        self.class_flow: np.array
        self.results = AssignmentResults()
        self.results.prepare(self.graph, self.matrix)
        self.results.reset()
        self._aon_results = AssignmentResults()
        self._aon_results.prepare(self.graph, self.matrix)

    def set_pce(self, pce: Union[float, int]) -> None:
        """Sets Passenger Car equivalent

        Args:
            pce (:obj:`Union[float, int]`): PCE. Defaults to 1 if not set
        """
        if not isinstance(pce, (float, int)):
            raise ValueError('PCE needs to be either integer or float ')
        self.pce = pce
