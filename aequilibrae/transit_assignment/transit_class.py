from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.assignment_class_base import TransportClassBase
from .assignment_results import TransitAssignmentResults
from .transit_graph import TransitGraph


class TransitClass(TransportClassBase):
    def __init__(self, name: str, graph: TransitGraph, matrix: AequilibraeMatrix):
        super().__init__(name, graph, matrix)
        self._config["Graph"] = str(graph._config)
        self.results = TransitAssignmentResults()

        if len(matrix.view_names) == 1:
            self.matrix_core = matrix.view_names[0]
        else:
            self.matrix_core = None

    def set_demand_matrix_core(self, core: str):
        """
        Set the matrix core to use for demand.

        :Arguments:
            **core** (:obj:`str`):"""
        if core not in self.matrix.view_names:
            raise KeyError(f"'{core}' is not present in `matrix.view_names`")
        self.matrix_core = core
