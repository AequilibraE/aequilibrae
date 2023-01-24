from typing import Union, List, Tuple, Dict
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

    def __init__(self, name: str, graph: Graph, matrix: AequilibraeMatrix) -> None:
        """
        Instantiates the class

         Args:
            name (:obj:`str`): UNIQUE class name.

            graph (:obj:`Graph`): Class/mode-specific graph

            matrix (:obj:`AequilibraeMatrix`): Class/mode-specific matrix. Supports multiple user classes
        """
        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

        if matrix.matrix_view.dtype != graph.default_types("float"):
            raise TypeError("Matrix's computational view need to be of type np.float64")

        self.graph = graph
        self.logger = graph.logger
        self.matrix = matrix
        self.pce = 1.0
        self.vot = 1.0
        self.mode = graph.mode
        self.class_flow: np.array
        self.results = AssignmentResults()
        self.fixed_cost = np.zeros(graph.graph.shape[0], graph.default_types("float"))
        self.fixed_cost_field = ""
        self.fc_multiplier = 1.0
        self._aon_results = AssignmentResults()
        self._selected_links = {}  # maps human name to link_set
        self.__id__ = name

    def set_pce(self, pce: Union[float, int]) -> None:
        """Sets Passenger Car equivalent

        Args:
            pce (:obj:`Union[float, int]`): PCE. Defaults to 1 if not set
        """
        if not isinstance(pce, (float, int)):
            raise ValueError("PCE needs to be either integer or float ")
        self.pce = pce

    def set_fixed_cost(self, field_name: str, multiplier=1):
        """Sets value of time

        Args:
            field_name (:obj:`str`): Name of the graph field with fixed costs for this class
            multiplier (:obj:`Union[float, int]`): Multiplier for the fixed cost. Defaults to 1 if not set
        """
        if field_name not in self.graph.graph.columns:
            raise ValueError("Field does not exist in the graph")

        self.fc_multiplier = float(multiplier)
        self.fixed_cost_field = field_name
        if np.any(np.isnan(self.graph.graph[field_name].values)):
            self.logger.warning(f"Cost field {field_name} has NaN values. Converted to zero")

        if self.graph.graph[field_name].min() < 0:
            msg = f"Cost field {field_name} has negative values. That is not allowed"
            self.logger.error(msg)
            raise ValueError(msg)

    def set_vot(self, value_of_time: float) -> None:
        """Sets value of time

        Args:
            value_of_time (:obj:`Union[float, int]`): Value of time. Defaults to 1 if not set
        """

        self.vot = float(value_of_time)

    def set_select_links(self, links: Dict[str, List[Tuple[int, int]]]):
        """Set the selected links. Checks if the links and directions are valid. Translates link_id and
        direction into unique link id used in compact graph.

        Args:
            links (:obj:`Link[Link[Tuple[int, int]]]`): Link IDs and directions to be used in select link analysis"""
        self._selected_links = {}
        for name, link_set in links.items():
            link_ids = []
            for link, dir in link_set:
                duplicate_link = False
                query = (self.graph.compact_graph["link_id"] == link) & (self.graph.compact_graph["direction"] == dir)
                if not query.any():
                    raise ValueError(f"link_id or direction {(link, dir)} is not present within graph.")
                # Check for duplicate compressed link ids in the current link set
                comp_id = self.graph.compact_graph[query]["id"].values[0]
                for val in link_ids:
                    if val == comp_id:
                        print(
                            f"Two input links map to the same compressed link in the network"
                            f", removing superfluous link {link}_{dir}")
                        duplicate_link = True
                        break
                if not duplicate_link:
                    link_ids.append(comp_id)
            self._selected_links[name] = tuple(set(link_ids))

    def __setattr__(self, key, value):

        if key not in [
            "graph",
            "logger",
            "matrix",
            "pce",
            "mode",
            "class_flow",
            "results",
            "_aon_results",
            "__id__",
            "vot",
            "fixed_cost",
            "fc_multiplier",
            "fixed_cost_field",
            "_selected_links",
        ]:
            raise KeyError("Traffic Class does not have that element")
        self.__dict__[key] = value
