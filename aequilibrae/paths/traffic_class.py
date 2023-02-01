from typing import Union, List, Tuple, Dict
import numpy as np
import pandas as pd

from aequilibrae.paths.graph import Graph
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.results import AssignmentResults
import warnings


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
        # self.sl_data = None

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

    def set_select_links(self, links: Union[None, Dict[str, List[Tuple[int, int]]]]):
        """Set the selected links. Checks if the links and directions are valid. Translates link_id and
        direction into unique link id used in compact graph.
        Supply links=None to disable select link analysis.

        Args:
            links (:obj:`Union[None, Dict[str, List[Tuple[int, int]]]]`): name of link set and
             Link IDs and directions to be used in select link analysis"""
        self._selected_links = {}
        if links is None:
            return

        for name, link_set in links.items():
            link_ids = []
            for link, dir in link_set:
                if dir == 0:
                    query = (self.graph.graph["link_id"] == link) & (
                        (self.graph.graph["direction"] == -1) | (self.graph.graph["direction"] == 1)
                    )
                else:
                    query = (self.graph.graph["link_id"] == link) & (self.graph.graph["direction"] == dir)
                    if not query.any():
                        raise ValueError(f"link_id or direction {(link, dir)} is not present within graph.")
                    # Check for duplicate compressed link ids in the current link set
                for comp_id in self.graph.graph[query]["__compressed_id__"].values:
                    if comp_id in link_ids:
                        warnings.warn(
                            "Two input links map to the same compressed link in the network"
                            f", removing superfluous link {link} and direction {dir} with compressed id {comp_id}"
                        )
                    else:
                        link_ids.append(comp_id)
            self._selected_links[name] = np.array(link_ids, dtype=self.graph.default_types("int"))
        # self.sl_data = links

    def decompress_select_link_flows(self) -> Dict[str, pd.DataFrame]:
        """
        Converts the select_link_flows from compressed link ids, back into regular ids.
        In addition, it maps the flow on the individual links based on their direction.

         Returns a dictionary of dataframes which map the link_set name to their link flows
         """
        decompressed_flows = {}
        num_subclasses = self.matrix.matrix_view.shape[2] if len(self.matrix.matrix_view.shape) > 2 else 1
        #Setting up common column names
        columns = []
        for x in range(num_subclasses):
            columns.append(f"ab_subclass_{x + 1}")
            columns.append(f"ba_subclass_{x + 1}")

        for name in self._selected_links.keys():
            link_loads = self.results.select_link_loading.matrix[name]

            n_links = self.graph.num_links
            graph = self.graph.graph
            ab_loading = np.zeros((n_links, num_subclasses))
            ba_loading = np.zeros((n_links, num_subclasses))

            for i, link in enumerate(link_loads):
                query = graph.query(f"__compressed_id__ == {i}")
                for j in query.itertuples():
                    # print(j)
                    if j.direction == 1:
                        # -1 to account for 0 indexing
                        ab_loading[j.link_id - 1, :] = link_loads[i, :]
                    else:
                        ba_loading[j.link_id - 1, :] = link_loads[i, :]
            loads = np.concatenate((ab_loading, ba_loading), axis=1)
            df = pd.DataFrame(loads, columns=sorted(columns))
            df.insert(loc=0, column="link_id", value=np.arange(1, self.graph.num_links + 1, 1))
            # rearrange columns so each subclass' flows are adjacent to each other
            decompressed_flows[name] = df[["link_id"] + columns]
        return decompressed_flows


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
