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

    def set_select_links(self, links: Dict[str, List[Tuple[int, int]]]):
        """Set the selected links. Checks if the links and directions are valid. Translates link_id and
        direction into unique link id used in compact graph.
        Supply links=None to disable select link analysis.

        Args:
            links (:obj:`Union[None, Dict[str, List[Tuple[int, int]]]]`): name of link set and
             Link IDs and directions to be used in select link analysis"""
        self._selected_links = {}
        for name, link_set in links.items():
            if len(name.split(" ")) != 1:
                warnings.warn("Input string name has a space in it. Replacing with _")
                name = str.join("_", name.split(" "))

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
        #Creating a column for each subclass in the TrafficClass for ab and ba flows
        num_subclasses = self.matrix.matrix_view.shape[2] if len(self.matrix.matrix_view.shape) > 2 else 1
        columns = []
        for x in range(num_subclasses):
            #TODO use the built in class names
            columns.append(f"ab_subclass_{x + 1}")
            columns.append(f"ba_subclass_{x + 1}")
        n_links = self.graph.num_links
        graph = self.graph.graph
        final_flows = {}
        # 3d array which stores ab flows in 0 index, ba flows in index 1
        # Within these indices, the flows for each subclass are stored in sequential order
        # e.g. subclass1_ab, subclass2_ab etc.
        sl_loading = np.empty((2, n_links, num_subclasses))
        for name in self._selected_links.keys():
    #TODO: RENAME to a nicer name
    #TODO: swap SL matrix to dictionary
            link_loads = self.results.select_link_loading.matrix[name]
            # CHANGE TO NANS/ RESET FLOW
            sl_loading.fill(0)
            # LAMBDA: Specifying how to use the map_links helper method
            func = lambda row: map_links(row["direction"] // 2, row["link_id"] - 1, row["__compressed_id__"],
                                         link_loads, sl_loading)
            graph.apply(func, axis=1)
            #turning np array of link flows into a df for writing into SQL
            loads = np.concatenate((sl_loading[0], sl_loading[1]), axis=1)
            # sorted(columns) ensure the order will have the subclass flows in the same order as the loads array
            # Specifically, ab_subclass1, ab_subclass2 ..., ba_subclass1, ba_subclass2, ...
            df = pd.DataFrame(loads, columns=sorted(columns))
            # Associate flows with their link ids
            df.insert(loc=0, column="link_id", value=np.arange(1, self.graph.num_links + 1, 1))
            # rearrange columns so each subclass' flows are adjacent to each other for user convenience
            # e.g. ab_subclass1, ba_subclass1, ab_subclass2 ...
            final_flows[name] = df[["link_id"] + columns]
        return final_flows

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

def map_links(dir: int, id: int , cid: int, links: np.array, res: np.array) -> None:
    """
    Helper method to decompress_select_link_flows. Takes an input direction, index (based on link_id),
    corresponding compressed link_od, compressed link flows and uncompressed link flow arrays.
    Maps the from the compressed array to the uncompressed array.
    """
    res[dir, id, :] = links[cid, :]
