import numpy as np
from aequilibrae.paths import Graph
from aequilibrae.matrix import AequilibraeMatrix as Matrix
from aequilibrae.paths.results import AssignmentResults


class AssignmentClass():
    def __init__(self, graph: Graph, matrix: Matrix) -> None:

        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

        self.graph = graph
        self.matrix = matrix
        self.class_flow: np.array

        self.results = AssignmentResults()
        self.results.prepare(self.graph, self.matrix)
        self.pce = 1

    def total_class_flow(self) -> None:
        self.class_flow = np.sum(self.results.link_loads, axis=1)

    def set_pce(self, pce: int) -> None:
        self.pce = pce
