from warnings import warn
from uuid import uuid4
from typing import Union
import numpy as np
from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.results import AssignmentResults


class TrafficClass:
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
        self._id = uuid4().hex
        self.__time_parameter__ = 1.0
        self.__fixed_cost_parameter__ = 0.0
        self.__fixed_cost_field__ = ""

    def set_pce(self, pce: Union[float, int]) -> None:
        """Sets Passenger Car equivalent

        Args:
            pce (:obj:`Union[float, int]`): PCE. Defaults to 1 if not set
        """
        if not isinstance(pce, (float, int)):
            raise ValueError("PCE needs to be either integer or float ")
        self.pce = pce

    def set_cost_function(self, fixed_cost_field="", time_cost_parameter=1.0, fixed_cost_parameter=0.0) -> None:
        """"Sets a generalized cost function for computing paths for this traffic class

        Args:
            time_cost_parameter (:obj:`float`): Parameter that multiplies time in the Generalized cost function
            fixed_cost_field (:obj:`str`): Field that contains the fixed (non congestion-related) portion of
            the cost for each link (must be field in the graph)
            fixed_cost_parameter (:obj:`float`): Parameter that multiplies the fixed portion of the cost in the
            Generalized cost function
        """

        if min(fixed_cost_parameter, time_cost_parameter) < 0:
            warn("Cost parameter can never be negative")
            return

        if time_cost_parameter == 0:
            warn("You made this class completely insensitive to congestion. Is that what you wanted?")

        if fixed_cost_field not in self.graph.graph.dtype.names:
            warn(f'Fixed cost must be one of [{",".join(self.graph.graph.dtype.names)}]')
            return

        self.__time_parameter__ = time_cost_parameter
        self.__fixed_cost_parameter__ = fixed_cost_parameter
        self.__fixed_cost_field__ = fixed_cost_field
