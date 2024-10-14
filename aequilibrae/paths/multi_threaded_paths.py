import numpy as np


class MultiThreadedPaths:
    def __init__(self):
        # The predecessors for each node in the graph
        self.predecessors = np.array([], np.int64)
        # Keeps the order in which the nodes were reached for the cascading network loading
        self.reached_first = np.array([], np.int64)
        # The previous link for each node in the tree
        self.connectors = np.array([], np.int64)
        #  holds the b_nodes in case of flows through centroid connectors are blocked
        self.temp_b_nodes = np.array([], np.int64)

    # In case we want to do by hand, we can prepare each method individually
    def prepare_(self, graph, results):
        itype = graph.default_types("int")
        self.predecessors = np.zeros((results.cores, results.nodes), dtype=itype)
        self.reached_first = np.zeros((results.cores, results.nodes), dtype=itype)
        self.connectors = np.zeros((results.cores, results.nodes), dtype=itype)
        self.temp_b_nodes = np.zeros((results.cores, graph.compact_graph.b_node.values.shape[0]), dtype=itype)

        for i in range(results.cores):
            self.temp_b_nodes[i, :] = graph.compact_graph.b_node.values[:]
