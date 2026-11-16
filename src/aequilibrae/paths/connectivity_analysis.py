import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

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
    """Identify nodes unable to reach and be reached from the main network.

    Blocked centroids can be endpoints, but cannot connect other nodes through
    themselves. The largest strongly connected component selects the main
    network; ties retain the first component. With no active links, the first
    retained node is the singleton main component.

    :Arguments:
        **anodes** (:obj:`np.ndarray`): Array of link tail node indices in graph order.
        **bnodes** (:obj:`np.ndarray`): Array of link head node indices in graph order.
        **all_nodes** (:obj:`np.ndarray`): Array of all retained node IDs in graph order.
        **block_centroid_flows** (:obj:`bool`): Whether to block flows through centroids.
        **centroids** (:obj:`np.ndarray`): Array of centroid node IDs.

    :Returns:
        :obj:`np.ndarray`: Array of disconnected node IDs.
    """
    if block_centroid_flows:
        if not len(all_nodes):
            return all_nodes.copy()

        # Find links entering and leaving each non-centroid node
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
        for node in set(in_links.keys()) & set(out_links.keys()):
            for in_link in in_links[node]:
                for out_link in out_links[node]:
                    if in_link != out_link:
                        turn_anodes.append(in_link)
                        turn_bnodes.append(out_link)

        reference = 0
        if anodes.size:
            disconnected = _analysis(
                np.array(turn_anodes, dtype=np.int64),
                np.array(turn_bnodes, dtype=np.int64),
                len(anodes),
            )
            main_links = np.ones(len(anodes), dtype=bool)
            main_links[disconnected] = False
            reference = int(anodes[np.flatnonzero(main_links)[0]])

        # Centroid connectors provide endpoint access, not evidence of disconnected tails.
        # Check both directions from the main component, stopping at blocked centroids.
        blocked = np.isin(all_nodes, centroids)
        blocked[reference] = False  # A centroid may still be the search origin.
        connected = np.ones(len(all_nodes), dtype=bool)
        for tails, heads in ((anodes, bnodes), (bnodes, anodes)):
            active = ~blocked[tails]
            csr = coo_matrix(
                (np.ones(np.count_nonzero(active)), (tails[active], heads[active])),
                shape=(len(all_nodes), len(all_nodes)),
            ).tocsr()
            reachable = np.zeros(len(all_nodes), dtype=bool)
            reachable[breadth_first_order(csr, reference, directed=True, return_predecessors=False)] = True
            connected &= reachable
        return all_nodes[~connected]
    else:
        # Non-blocking: find nodes in largest strongly connected component
        disconnected_indices = _analysis(anodes, bnodes, len(all_nodes))
        return all_nodes[disconnected_indices]
