import numpy as np
import tables

PATH_FILE_COMPRESSION = tables.Filters(complevel=1, complib="blosc:zstd", shuffle=True)


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

        if results.save_path_file:
            # Keep output rows separate from per-thread routing scratch. Otherwise
            # thread 0 overwrites the tree stored for origin 0 on its next origin.
            rows = graph.num_zones + results.cores
            self.predecessors = np.zeros((rows, results.compact_nodes), dtype=itype)
            self.connectors = np.zeros((rows, results.compact_nodes), dtype=itype)
        else:
            self.predecessors = np.zeros((results.cores, results.compact_nodes), dtype=itype)
            self.connectors = np.zeros((results.cores, results.compact_nodes), dtype=itype)

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
        self.temp_link_loads = np.zeros((results.cores, results.links + 1, results.classes["number"]), dtype=ftype)
        self.temp_b_nodes = np.zeros((results.cores, compact_b_nodes.shape[0]), dtype=itype)

        for i in range(results.cores):
            self.temp_b_nodes[i, :] = compact_b_nodes[:]

    def save_path_files(self, path: str, graph, iteration: int):
        """
        Expand a compressed-graph shortest-path tree back to the full (uncompressed) network then save it.
        """

        a_nodes = graph.graph["a_node"].to_numpy()
        b_nodes = graph.graph["b_node"].to_numpy()

        mapping_idx, mapping_data, _ = graph.create_compressed_link_network_mapping()
        # The compression mapping stores stable supernetwork IDs, while path
        # serialization indexes the current graph row order.
        supernet_ids = graph.graph["__supernet_id__"].to_numpy()
        row_by_supernet = np.empty(graph.num_links, dtype=np.int64)
        row_by_supernet[supernet_ids] = np.arange(graph.num_links)
        counts = np.diff(mapping_idx).astype(int)

        all_preds = self.predecessors
        all_conns = self.connectors
        num_origins = graph.num_zones
        num_network_nodes = len(graph.all_nodes)

        # Scratch buffers reused per origin
        network_predecessors = np.zeros(num_network_nodes, dtype=np.int32)
        network_connectors = np.zeros(num_network_nodes, dtype=np.int32)

        with tables.open_file(path, mode="a", title="Predecessor Trees") as h5:
            # Group for this iteration
            grp = h5.create_group("/", f"iteration_{iteration}", f"Assignment iteration {iteration}")

            carr_preds = h5.create_carray(
                grp,
                "predecessors",
                tables.Int32Atom(),
                shape=(num_origins, num_network_nodes),
                chunkshape=(1, num_network_nodes),  # write & read one origin at a time
                filters=PATH_FILE_COMPRESSION,
            )
            carr_conns = h5.create_carray(
                grp,
                "connectors",
                tables.Int32Atom(),
                shape=(num_origins, num_network_nodes),
                chunkshape=(1, num_network_nodes),
                filters=PATH_FILE_COMPRESSION,
            )

            # Expand & write one origin per row
            for origin in range(num_origins):
                network_predecessors.fill(-1)
                network_connectors.fill(-1)

                preds = all_preds[origin]
                conns = all_conns[origin]

                # Which compact nodes / compressed links are used?
                valid = (preds != -1) & (conns >= 0)
                used_clinks = np.unique(conns[valid])

                is_used = np.zeros(len(mapping_idx) - 1, dtype=bool)
                is_used[used_clinks] = True

                expanded_mask = np.repeat(is_used, counts)
                graph_idxs = row_by_supernet[mapping_data[expanded_mask]]

                tail = b_nodes[graph_idxs]
                head = a_nodes[graph_idxs]

                network_predecessors[tail] = head
                network_connectors[tail] = graph_idxs

                # Write this origin's row into the HDF5 carray
                carr_preds[origin, :] = network_predecessors
                carr_conns[origin, :] = network_connectors

            h5.set_node_attr(grp, "num_origins", num_origins)
            h5.set_node_attr(grp, "num_network_nodes", num_network_nodes)
            h5.set_node_attr(grp, "iteration", iteration)
