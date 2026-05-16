import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.dont_write_bytecode = True


def _analysis(anodes: np.ndarray, bnodes: np.ndarray) -> np.ndarray:
    n = np.max([np.max(anodes), np.max(bnodes)]) + 1
    csr = coo_matrix((np.ones(anodes.shape[0]), (anodes, bnodes)), shape=(n, n)).tocsr()
    n_components, labels = connected_components(csgraph=csr, directed=True, return_labels=True, connection="strong")

    # We then identify all the link/directions that have the highest connectivity degree (i.e. the biggest island)
    bc = np.bincount(labels)
    max_label = np.where(bc == bc.max())[0][0]
    return np.where(labels != max_label)[0]


def non_blocking_through_centroids(graph) -> np.ndarray:
    anodes = graph.graph.a_node.to_numpy()
    bnodes = graph.graph.b_node.to_numpy()
    disconnected = _analysis(anodes, bnodes)
    return graph.all_nodes[disconnected]


def blocking_through_centroids(graph) -> np.ndarray:
    edges = graph.graph

    turns = (
        edges[["id", "b_node"]].rename(columns={"id": "in_link", "b_node": "node"})
        .merge(
            edges[["id", "a_node"]].rename(columns={"id": "out_link", "a_node": "node"}),
            on="node",
            how="inner",
        )
    )

    turns = turns[turns["in_link"] != turns["out_link"]]
    turns = turns[turns["node"] < graph.centroids.shape[0]]

    anodes = turns.in_link.to_numpy()
    bnodes = turns.out_link.to_numpy()

    disconnected = _analysis(anodes, bnodes)
    if disconnected.shape[0] == 0:
        return disconnected
    disconnected = graph.graph[graph.graph["id"].isin(disconnected)]["a_node"].to_numpy()
    return graph.all_nodes[disconnected]


def disconnected_analysis(graph) -> np.ndarray:
    if graph.block_centroid_flows:
        return blocking_through_centroids(graph)
    return non_blocking_through_centroids(graph)
