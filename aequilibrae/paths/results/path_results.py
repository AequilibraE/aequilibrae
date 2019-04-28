import numpy as np
from ..graph import Graph
from ..AoN import update_path_trace


class PathResults:
    def __init__(self):
        """
        @type graph: Set of numpy arrays to store Computation results
        """
        self.predecessors = None
        self.connectors = None
        self.skims = None
        self.path = None
        self.path_nodes = None
        self.milepost = None
        self.reached_first = None
        self.origin = None
        self.destination = None

        self.links = -1
        self.nodes = -1
        self.zones = -1
        self.num_skims = -1
        self.__integer_type = None
        self.__float_type = None
        self.__graph_id__ = None

    def prepare(self, graph):
        self.__integer_type = graph.default_types("int")
        self.__float_type = graph.default_types("float")
        self.nodes = graph.num_nodes + 1
        self.zones = graph.centroids + 1
        self.links = graph.num_links + 1
        self.num_skims = graph.skims.shape[1]

        self.predecessors = np.zeros(self.nodes, dtype=self.__integer_type)
        self.connectors = np.zeros(self.nodes, dtype=self.__integer_type)
        self.reached_first = np.zeros(self.nodes, dtype=self.__integer_type)
        self.skims = np.zeros((self.nodes, self.num_skims), self.__float_type)
        self.__graph_id__ = graph.__id__

    def reset(self):
        if self.predecessors is not None:
            self.predecessors.fill(-1)
            self.connectors.fill(-1)
            self.skims.fill(np.inf)
            self.path = None
            self.path_nodes = None
            self.milepost = None

        else:
            raise ValueError(
                "Exception: Path results object was not yet prepared/initialized"
            )

    def update_trace(self, graph, destination):
        # type: (Graph, int) -> (None)
        if not isinstance(destination, int):
            raise TypeError("destination needs to be an integer")

        if not isinstance(graph, Graph):
            raise TypeError("graph needs to be an AequilibraE Graph")

        if destination >= graph.nodes_to_indices.shape[0]:
            raise ValueError(
                "destination out of the range of node numbers in the graph"
            )

        if self.__graph_id__ != graph.__id__:
            raise ValueError(
                "Results object not prepared. Use --> results.prepare(graph)"
            )

        update_path_trace(self, destination, graph)
