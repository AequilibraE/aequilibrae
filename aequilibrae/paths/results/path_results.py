import numpy as np

from aequilibrae import global_logger
from aequilibrae.paths.graph import Graph
from typing import Union, List

try:
    from aequilibrae.paths.AoN import update_path_trace, path_computation, HEURISTIC_MAP
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")


class PathResults:
    """Path computation result holder

    .. code-block:: python

          >>> from aequilibrae import Project
          >>> from aequilibrae.paths.results import PathResults

          >>> proj = Project.from_path("/tmp/test_project")
          >>> proj.network.build_graphs()

          # Mode c is car in this project
          >>> car_graph = proj.network.graphs['c']

          # minimize distance
          >>> car_graph.set_graph('distance')

          # If you want to compute skims
          # It does increase path computation time substantially
          >>> car_graph.set_skimming(['distance', 'free_flow_time'])

          >>> res = PathResults()
          >>> res.prepare(car_graph)
          >>> res.compute_path(1, 17)

          # res.milepost contains the milepost corresponding to each node along the path
          # res.path_nodes contains the sequence of nodes that form the path
          # res.path  contains the sequence of links that form the path
          # res.path_link_directions contains the link directions corresponding to the above links
          # res.skims contain all skims requested when preparing the graph

          # Update all the outputs mentioned above for destination 9. Same origin: 1
          >>> res.update_trace(9)

          # clears all computation results
          >>> res.reset()
    """

    def __init__(self) -> None:
        self.predecessors = None
        self.connectors = None
        self.skims = None
        self._skimming_array = None
        self.path = None
        self.path_nodes = None
        self.path_link_directions = None
        self.milepost = None
        self.reached_first = None
        self.origin = None
        self.destination = None
        self.graph: Graph = None
        self.early_exit = False
        self.a_star = False
        self.links = -1
        self.nodes = -1
        self.zones = -1
        self.num_skims = -1
        self.__integer_type = None
        self.__float_type = None
        self.__graph_id__ = None
        self.__graph_sum = None
        self._early_exit = self.early_exit
        self._a_star = self.a_star
        self._heuristic = "equirectangular"

    def compute_path(
        self,
        origin: int,
        destination: int,
        early_exit: bool = False,
        a_star: bool = False,
        heuristic: Union[str, None] = None,
    ) -> None:
        """
        Computes the path between two nodes in the network.

        A* heuristics are currently only valid distance cost fields.

        :Arguments:
            **origin** (:obj:`int`): Origin for the path

            **destination** (:obj:`int`): Destination for the path

            **early_exit** (:obj:`bool`): Stop constructing the shortest path tree once the destination is found.
                                          Doing so may cause subsequent calls to `update_trace` to recompute the tree.
                                          Default is `False`.
            **a_star** (:obj:`bool`): Whether or not to use A* over Dijkstra's algorithm. When `True`, `early_exit`
                                      is always `True`. Default is `False`.
            **heuristic** (:obj:`str`): Heuristic to use if `a_star` is enabled. Default is `None`.
        """

        if self.graph is None:
            raise Exception("You need to set graph skimming before you compute a path")

        if a_star and self.graph.lonlat_index.empty:
            raise Exception("You need to supply a lon/lat index to graph.prepare_graph to use A*")

        self.early_exit = self._early_exit = early_exit or a_star
        self.a_star = self._a_star = a_star
        if heuristic is not None:
            self.set_heuristic(heuristic)
        path_computation(origin, destination, self.graph, self)
        if self.graph.skim_fields:
            self.skims.fill(np.inf)
            self.skims[self.graph.all_nodes, :] = self._skimming_array[:-1, :]
            self.skims[self.skims > self.__graph_sum] = np.inf

    def prepare(self, graph: Graph) -> None:
        """
        Prepares the object with dimensions corresponding to the graph object

        :Arguments:
            **graph** (:obj:`Graph`): Needs to have been set with number of centroids and list of skims (if any)
        """

        if not graph.cost_field:
            raise Exception('Cost field needs to be set for cost computation. use graph.set_graph("your_cost_field")')

        self.__integer_type = graph.default_types("int")
        self.__float_type = graph.default_types("float")
        self.nodes = graph.num_nodes + 1
        self.zones = graph.centroids + 1
        self.links = graph.num_links + 1
        self.num_skims = len(graph.skim_fields)

        self.predecessors = np.zeros(self.nodes, dtype=self.__integer_type)
        self.connectors = np.zeros(self.nodes, dtype=self.__integer_type)
        self.reached_first = np.zeros(self.nodes, dtype=self.__integer_type)
        if self.num_skims:
            self.skims = np.empty((np.max(graph.all_nodes) + 1, self.num_skims), self.__float_type)
            self.skims.fill(np.inf)
            self._skimming_array = np.zeros((self.nodes, self.num_skims), self.__float_type)
        else:
            self._skimming_array = np.zeros((1, 2), self.__float_type)

        self.__graph_id__ = graph.__id__
        # We can imagine somebody creating a worst-case scenario network (imagining that turn penalties are considered)
        # where one needs to traverse all links (or almost all) in both directions.
        self.__graph_sum = 2 * graph.cost.sum()
        self.graph = graph

    def reset(self) -> None:
        """
        Resets object to prepared and pre-computation state
        """
        if self.predecessors is not None:
            self.predecessors.fill(-1)
            self.connectors.fill(-1)
            if self.skims is not None:
                self.skims.fill(np.inf)
            self._skimming_array.fill(np.inf)
            self.path = None
            self.path_nodes = None
            self.path_link_directions = None
            self.milepost = None
            self._early_exit = self.early_exit = False
            self._a_star = self.a_star = False
            self._heuristic = "equirectangular"

        else:
            raise ValueError("Exception: Path results object was not yet prepared/initialized")

    def update_trace(self, destination: int) -> None:
        """
        Updates the path's nodes, links, skims and mileposts

        If the previously computed path had `early_exit` enabled, `update_trace` will check if the
        `destination` has already been found, if not the shortest path tree will be recomputed with the `early_exit`
        argument passed on.

        If the previously computed path had `a_star` enabled, `update_trace` always recompute the path.

        :Arguments:
            **destination** (:obj:`int`): ID of the node we are computing the path too
        """
        if not isinstance(destination, int):
            raise TypeError("destination needs to be an integer")

        if destination >= self.graph.nodes_to_indices.shape[0]:
            raise ValueError("destination out of the range of node numbers in the graph")

        update_path_trace(self, destination, self.graph)

    def set_heuristic(self, heuristic: str) -> None:
        """
        Set the heuristics to be used in A*. Must be one of `get_heuristics()`.

        :Arguments:
            **heuristic** (:obj:`str`): Heuristic to use in A*.
        """
        if heuristic not in HEURISTIC_MAP.keys():
            raise ValueError(f"heruistic must be one of {self.get_heuristics()}")

        self._heuristic = heuristic

    def get_heuristics(self) -> List[str]:
        """Return the availiable heuristics."""
        return list(HEURISTIC_MAP.keys())
