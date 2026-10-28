import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.dont_write_bytecode = True


def _analysis(anodes: np.ndarray, bnodes: np.ndarray, num_nodes: int = 0) -> np.ndarray:
    if (anodes.size == 0 or bnodes.size == 0) and not num_nodes:
        return np.array([], dtype=np.int64)

    n = max(int(max(anodes.max(), bnodes.max())) + 1 if anodes.size else 0, num_nodes)
    csr = coo_matrix((np.ones(anodes.shape[0]), (anodes, bnodes)), shape=(n, n)).tocsr()
    n_components, labels = connected_components(csgraph=csr, directed=True, return_labels=True, connection="strong")

    # We then identify all the link/directions that have the highest connectivity degree (i.e. the biggest island)
    bc = np.bincount(labels)
    max_label = np.where(bc == bc.max())[0][0]
    return np.where(labels != max_label)[0]


def disconnected_analysis(
    anodes: np.ndarray,
    bnodes: np.ndarray,
    all_nodes: np.ndarray,
    block_centroid_flows: bool,
    centroids: np.ndarray,
) -> np.ndarray:
    """Identify disconnected nodes using strongly connected components analysis.

    Args:
        anodes: Array of link tail node indices (in compact routing order)
        bnodes: Array of link head node indices (in compact routing order)
        all_nodes: Array of all node IDs in network order
        block_centroid_flows: Whether to block flows through centroids
        centroids: Array of centroid node IDs

    Returns:
        Array of disconnected node IDs
    """
    if block_centroid_flows:
        # Turn restrictions: find links entering and leaving each non-centroid node
        nodes_to_indices = np.full(int(all_nodes.max()) + 1, -1, dtype=np.int64)
        nodes_to_indices[all_nodes] = np.arange(len(all_nodes))
        centroid_set = set(centroids)

        # Find which compact links form turn transitions through non-centroid nodes
        in_links = {}
        out_links = {}
        for i, (a, b) in enumerate(zip(anodes, bnodes, strict=True)):
            node = all_nodes[int(b)]
            if node not in centroid_set:
                out_links.setdefault(node, []).append(i)
            node = all_nodes[int(a)]
            if node not in centroid_set:
                in_links.setdefault(node, []).append(i)

        # Build transition graph: links that can follow each link
        turn_anodes = []
        turn_bnodes = []
        for node in (set(in_links.keys()) & set(out_links.keys())):
            for in_link in in_links[node]:
                for out_link in out_links[node]:
                    if in_link != out_link:
                        turn_anodes.append(in_link)
                        turn_bnodes.append(out_link)

        if turn_anodes:
            disconnected = _analysis(
                np.array(turn_anodes, dtype=np.int64),
                np.array(turn_bnodes, dtype=np.int64),
            )
            if disconnected.size > 0:
                disconnected_links = disconnected
                # Map back to nodes
                disconnected_nodes = np.unique(all_nodes[anodes[disconnected_links]])
                return all_nodes[np.isin(all_nodes, disconnected_nodes)]
        return np.array([], dtype=np.int64)
    else:
        # Non-blocking: find nodes in largest strongly connected component
        disconnected_indices = _analysis(anodes, bnodes, len(all_nodes))
        return all_nodes[disconnected_indices]
