from typing import Union
import numpy as np
from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix as Matrix
from aequilibrae.paths.results import AssignmentResults


class TrafficClass():
    def __init__(self, graph: Graph, matrix: Matrix) -> None:
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
        if not isinstance(pce, (float, int)):
            raise ValueError('PCE needs to be either integer or float ')
        self.pce = pce
