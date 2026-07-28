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
        #  holds the b_nodes in case of flows through centroid connectors are blocked
        self.temp_b_nodes = np.array([])
        # Temporary array which stores whether a link is accessed in a path for Select Link Analysis functionality
        self.has_flow_mask = np.array([])
        # Stores all selected link sets in one array
        self.select_links = np.array([])
        # Stores all select link OD matrices
        self.temp_sl_od_matrix = np.array([])
        # Stores all link loading matrices
        self.temp_sl_link_loading = np.array([])
        # Maps the names of the SL link sets to array indices
        self.sl_idx = {}
        # Arc predecessors for turn restrictions (arc-based path finding)
        self.arc_predecessors = np.array([])
        # Node-level turn penalties for arc-based skimming
        self.node_turn_penalties = np.array([])
        # Arc-level turn penalties for network loading (path reconstruction)
        self.arc_turn_penalties = np.array([])
        # Total turn penalty accumulator (per thread)
        self.turn_penalty_accumulator = np.array([])
        # Arc-based node label costs (per thread)
        self.node_label_costs = np.array([])

    # In case we want to do by hand, we can prepare each method individually

    def prepare(self, graph, results):
        itype = graph.default_types("int")
        ftype = graph.default_types("float")
        compact_b_nodes = graph.compact_graph.b_node.to_numpy(copy=False)
        self.predecessors = np.zeros((results.cores, results.compact_nodes), dtype=itype)

        # Allocate arc predecessors if turn restrictions are present
        size_nodes = results.compact_nodes if graph.has_turn_restrictions else 1
        self.node_label_costs = np.zeros((results.cores, size_nodes), dtype=ftype)
        self.node_turn_penalties = np.zeros((results.cores, size_nodes), dtype=ftype)

        self.turn_penalty_accumulator = np.zeros(results.cores, dtype=ftype)

        size_links = graph.compact_num_links + 1 if graph.has_turn_restrictions else 1
        self.arc_predecessors = np.zeros((results.cores, size_links), dtype=itype)
        self.arc_turn_penalties = np.zeros((results.cores, size_links), dtype=ftype)

        if results._selected_links:
            self.has_flow_mask = np.zeros((results.cores, graph.compact_num_links), dtype=bool)
            # Copying the select link matrices from results
            self.select_links = results.select_links
            self.temp_sl_od_matrix = np.zeros(
                (
                    results.cores,
                    len(results._selected_links),
                    graph.num_zones,
                    graph.num_zones,
                    results.classes["number"],
                ),
                dtype=graph.default_types("float"),
            )
            self.temp_sl_link_loading = np.zeros(
                (results.cores, len(results._selected_links), graph.compact_num_links, results.classes["number"]),
                dtype=graph.default_types("float"),
            )

        if results.num_skims > 0:
            self.temporary_skims = np.zeros((results.cores, results.compact_nodes, results.num_skims), dtype=ftype)
        else:
            self.temporary_skims = np.zeros((results.cores, 1, 1), dtype=ftype)
        self.reached_first = np.zeros((results.cores, results.compact_nodes), dtype=itype)
        self.connectors = np.zeros((results.cores, results.compact_nodes), dtype=itype)
        self.temp_link_loads = np.zeros((results.cores, results.links + 1, results.classes["number"]), dtype=ftype)
        self.temp_b_nodes = np.zeros((results.cores, compact_b_nodes.shape[0]), dtype=itype)

        for i in range(results.cores):
            self.temp_b_nodes[i, :] = compact_b_nodes[:]
