import sys
from collections import deque

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from aequilibrae.utils.aeq_signal import SIGNAL

sys.dont_write_bytecode = True


def _build_adjacency(graph, origin_index=None) -> csr_matrix:
    fs = graph.compact_fs
    base_b_nodes = graph.compact_graph.b_node.to_numpy(copy=False)
    b_nodes = base_b_nodes if origin_index is None else np.array(base_b_nodes, copy=True)

    if origin_index is not None:
        # Replicates the centroid-blocking behavior used in path-finding: all centroid connectors
        # point back to the origin centroid except the origin's own outgoing connectors.
        zones = graph.num_zones
        b_nodes[: fs[zones]] = origin_index
        b_nodes[fs[origin_index] : fs[origin_index + 1]] = base_b_nodes[fs[origin_index] : fs[origin_index + 1]]

    node_count = fs.shape[0] - 1
    return csr_matrix((np.ones(b_nodes.shape[0], dtype=np.int8), b_nodes, fs), shape=(node_count, node_count))


def _component_reachability(adjacency: csr_matrix):
    num_components, labels = connected_components(adjacency, directed=True, connection="strong", return_labels=True)

    children = [set() for _ in range(num_components)]
    coo = adjacency.tocoo(copy=False)
    for source, target in zip(labels[coo.row], labels[coo.col]):
        if source != target:
            children[source].add(target)

    reachable = []
    for component in range(num_components):
        seen = {component}
        queue = deque([component])
        while queue:
            current = queue.popleft()
            for nxt in children[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        reachable.append(seen)
    return labels, reachable


def _disconnected_pairs(graph, origins=None, signal=None) -> pd.DataFrame:
    origins = np.asarray(origins if origins is not None else graph.centroids, dtype=graph.centroids.dtype)
    centroid_nodes = graph.compact_nodes_to_indices[graph.centroids]
    records = []
    precomputed = None

    if not graph.block_centroid_flows:
        precomputed = _component_reachability(_build_adjacency(graph))

    total = origins.shape[0]
    for position, origin in enumerate(origins, start=1):
        origin_index = int(graph.compact_nodes_to_indices[origin])
        if origin_index < 0:
            raise ValueError(f"Origin {origin} is not present in the graph")

        if graph.block_centroid_flows:
            labels, reachable = _component_reachability(_build_adjacency(graph, origin_index))
        else:
            if precomputed is None:
                raise RuntimeError("Precomputed connectivity labels are unavailable")
            labels, reachable = precomputed

        allowed = reachable[labels[origin_index]]
        disconnected = [
            int(destination)
            for destination_index, destination in zip(centroid_nodes, graph.centroids)
            if destination != origin and labels[destination_index] not in allowed
        ]
        records.extend((int(origin), destination) for destination in disconnected)

        if signal is not None:
            signal.emit(["zones finalized", position])
            signal.emit(["text connectivity", f"{position} / {total}"])

    return pd.DataFrame(records, columns=["origin", "destination"], dtype=np.int64)


class ConnectivityAnalysis:
    """

    .. code-block:: python

        >>> from aequilibrae.paths.connectivity_analysis import ConnectivityAnalysis

        >>> project = create_example(project_path)

        >>> network = project.network
        >>> network.build_graphs()

        >>> graph = network.graphs['c']
        >>> graph.set_graph(cost_field="distance")
        >>> graph.set_blocked_centroid_flows(False)

        >>> conn_test = ConnectivityAnalysis(graph)
        >>> conn_test.execute()

        # The connectivity tester report as a Pandas DataFrame
        >>> disconnected = conn_test.disconnected_pairs

        >>> project.close()
    """

    connectivity = SIGNAL(object)

    def __init__(self, graph, origins=None, project=None):
        self.project = project
        self.origins = origins
        self.graph = graph
        self.report = []
        self.procedure_id = ""
        self.procedure_date = ""
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        """Runs the skimming process as specified in the graph"""

        self.disconnected_pairs = _disconnected_pairs(self.graph, self.origins, self.connectivity)
        self.disconnected_pairs = self.disconnected_pairs.sort_values(["origin", "destination"])
