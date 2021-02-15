from uuid import uuid4
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

        if matrix.matrix_view.dtype != graph.default_types('float'):
            raise TypeError("Matrix's computational view need to be of type np.float64")

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
        self.__id__ = uuid4().hex

    def set_pce(self, pce: Union[float, int]) -> None:
        """Sets Passenger Car equivalent

        Args:
            pce (:obj:`Union[float, int]`): PCE. Defaults to 1 if not set
        """
        if not isinstance(pce, (float, int)):
            raise ValueError('PCE needs to be either integer or float ')
        self.pce = pce

    def __setattr__(self, key, value):

        if key not in ['graph', 'matrix', 'pce', 'mode', 'class_flow', 'results', '_aon_results', '__id__']:
            raise KeyError('Traffic Class does not have that element')
        self.__dict__[key] = value
