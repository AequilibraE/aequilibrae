import numpy as np


class MultiThreadedAoN:
    def __init__(self):
        # The predecessors for each node in the graph
        self.predecessors = np.array([])
        # holds the skims for all nodes in the network (during path finding)
        self.temporary_skims = np.array([])
        # Keeps the order in which the nodes were reached for the cascading network loading
        self.reached_first = np.array([])
        # The previous link for each node in the tree
        self.connectors = np.array([])
        # Temporary results for assignment. Necessary for parallelization
        self.temp_link_loads = np.array([])
        # Temporary nodes for assignment. Necessary for cascading
        self.temp_node_loads = np.array([])
        #  holds the b_nodes in case of flows through centroid connectors are blocked
        self.temp_b_nodes = np.array([])

    # In case we want to do by hand, we can prepare each method individually
    def prepare(self, graph, results):
        itype = graph.default_types("int")
        ftype = graph.default_types("float")
        self.predecessors = np.zeros((results.compact_nodes, results.cores), dtype=itype)
        if results.num_skims > 0:
            self.temporary_skims = np.zeros((results.compact_nodes, results.num_skims, results.cores), dtype=ftype)
        else:
            self.temporary_skims = np.zeros((1, 1, results.cores), dtype=ftype)
        self.reached_first = np.zeros((results.compact_nodes, results.cores), dtype=itype)
        self.connectors = np.zeros((results.compact_nodes, results.cores), dtype=itype)
        self.temp_link_loads = np.zeros((results.links + 1, results.classes["number"], results.cores), dtype=ftype)
        self.temp_node_loads = np.zeros((results.compact_nodes, results.classes["number"], results.cores), dtype=ftype)
        self.temp_b_nodes = np.zeros((graph.compact_graph.b_node.shape[0], results.cores), dtype=itype)
        for i in range(results.cores):
            self.temp_b_nodes[:, i] = graph.compact_graph.b_node.values[:]
