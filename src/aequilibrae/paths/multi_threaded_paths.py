import numpy as np


class MultiThreadedPaths:
    def __init__(self, graph, cores, nodes):
        itype = graph.default_types("int")
        compact_b_nodes = graph.compact_graph.b_node.to_numpy(copy=False)
        self.predecessors = np.zeros((cores, nodes), dtype=itype)
        self.reached_first = np.zeros((cores, nodes), dtype=itype)
        self.connectors = np.zeros((cores, nodes), dtype=itype)
        self.temp_b_nodes = np.zeros((cores, compact_b_nodes.shape[0]), dtype=itype)
        self.destinations = np.zeros((cores, nodes), dtype=bool)

        centroids_idx = graph.nodes_to_indices[graph.centroids]
        for i in range(cores):
            self.temp_b_nodes[i, :] = compact_b_nodes
            self.destinations[i, centroids_idx] = True
