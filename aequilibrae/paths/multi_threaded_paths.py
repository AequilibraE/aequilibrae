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
    def prepare_(self, graph, cores, nodes):
        itype = graph.default_types("int")
        self.predecessors = np.zeros((cores, nodes), dtype=itype)
        self.reached_first = np.zeros((cores, nodes), dtype=itype)
        self.connectors = np.zeros((cores, nodes), dtype=itype)
        self.temp_b_nodes = np.zeros((cores, graph.compact_graph.b_node.values.shape[0]), dtype=itype)
        self.destinations = np.zeros((cores, nodes), dtype=bool)

        centroids_idx = graph.nodes_to_indices[graph.centroids]
        for i in range(cores):
            self.temp_b_nodes[i, :] = graph.compact_graph.b_node.values[:]
            self.destinations[i, centroids_idx] = True
