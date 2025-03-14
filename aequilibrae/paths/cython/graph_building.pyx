import pandas as pd
import numpy as np
cimport numpy as np
cimport cython

from libcpp.queue cimport queue


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef void _remove_dead_ends(
    long long [:] graph_fs,
    long long [:] all_nodes,
    long long [:] nodes_to_indices,
    long long [:] a_nodes,
    long long [:] b_nodes,
    signed char [:] directions,
    long long [:] in_degree,
    long long [:] out_degree,
    np.uint8_t [:] burnt_links,
) noexcept nogil:
    cdef:
        long long b_node
        Py_ssize_t node_idx, incoming

        queue[long long] Q

    # Discovery: we are looking for potential dead ends, this includes:
    #     - nodes with all incoming edges,
    #     - nodes with all outgoing edges,
    #     - nodes for which all out going links point to the same node and has the same number of back links.
    #       This is a generalisation of dead-ends formed by a node with a single bidirectional link to another
    #
    # Removal: At nodes of interest we begin a Breadth First Search (BFS) for links we can remove. It's uncommon for the
    # BFS to extend far as the stopping criteria and conditions for expansion are very similar, often this just searches
    # along a line.  For removal the node must have "no choices", that is, all outgoing edges point to the same node
    # *and* all incoming edges can be account for as from that same node.
    # The criteria for expansion is that there are no incoming edges *and* all outgoing edges point to the same node.
    for starting_node_idx in range(all_nodes.shape[0]):
        Q.push(starting_node_idx)
        while not Q.empty():
            node_idx = Q.front()
            Q.pop()

            # Centroids are marked with negative degrees, the actual degree does not matter
            if in_degree[node_idx] < 0 or out_degree[node_idx] < 0:
                continue
            elif in_degree[node_idx] == 0 and out_degree[node_idx] == 0:
                continue
            elif in_degree[node_idx] > 0 and out_degree[node_idx] == 0:
                # All incoming or all outgoing edges, since there's no way to either leave, or get to this node all the
                # attached edges are of no use. However we have no (current) means to remove these edges, a transpose of
                # the graph would be required to avoid individual lookups.
                continue

            # ### Expansion
            if in_degree[node_idx] == 0 and out_degree[node_idx] > 0:
                for link_idx in range(graph_fs[node_idx], graph_fs[node_idx + 1]):
                    if not burnt_links[link_idx]:
                        # Only need to burn links leaving this node
                        burnt_links[link_idx] = True
                        in_degree[b_nodes[link_idx]] -= 1
                        out_degree[node_idx] -= 1
                        Q.push(b_nodes[link_idx])
                continue

            # ### Propagation
            # We now know that the node we are looking at has a mix of incoming and outgoing edges, i.e. in_degree[node]
            # > 0 and out_degree[node] > 0 That implies that this node is reachable from some other node. We now need to
            # assess if this node would ever be considered in pathfinding.  To be considered, there needs to be some
            # form of real choice, i.e. there needs to be multiple ways to leave a new node. If the only way to leave
            # this node is back the way we came then the cost of the path
            #     ... -> pre -> node -> pre -> ...
            # is greater than that of
            #     ... -> pre -> ...
            # so we can remove this link to node. This is because negative cost cycles are disallowed in path finding.
            #
            # We don't remove the case where there is only one new node that could be used to leave as it is handled in
            # the graph compression.  It would however, be possible to handle that here as well.

            # If all the outgoing edges are to the same node *and* all incoming edges come from that node, then there is
            # no real choice and the link should be removed. This is primarily to catch nodes who are only connected to
            # a single other node via a bidirectional link but can handle the case where multiple links point between
            # the same pair of nodes with perhaps, different, but still non-negative costs.

            # Lets first find a link that isn't burnt. Then check that every link after this one points to the same node
            b_node = -1
            for link_idx in range(graph_fs[node_idx], graph_fs[node_idx + 1]):
                if burnt_links[link_idx]:
                    continue

                if b_node == -1:
                    b_node = b_nodes[link_idx]
                elif b_node != b_nodes[link_idx]:
                    # We've found a link to a different node, that means we have some other way to leave and we can't
                    # remove this node.
                    break
            else:
                # We now know all outgoing edges point to the same node. Lets now check that all incoming edges are
                # accounted for
                incoming = in_degree[node_idx]
                for link_idx in range(graph_fs[b_node], graph_fs[b_node + 1]):
                    # Incoming degree has already been decremented, when a link was burnt
                    if not burnt_links[link_idx] and b_nodes[link_idx] == node_idx:
                        incoming -= 1

                if incoming != 0:
                    # Not all incoming edges are accounted for, there is some other node that links to this one. From
                    # where we don't know.
                    continue

                # All incoming edges are accounted for, this node is of interest.
                for link_idx in range(graph_fs[node_idx], graph_fs[node_idx + 1]):
                    if not burnt_links[link_idx]:
                        burnt_links[link_idx] = True
                        in_degree[b_node] -= 1
                        out_degree[node_idx] -= 1

                for link_idx in range(graph_fs[b_node], graph_fs[b_node + 1]):
                    if not burnt_links[link_idx] and b_nodes[link_idx] == node_idx:
                        burnt_links[link_idx] = True
                        in_degree[node_idx] -= 1
                        out_degree[b_node] -= 1

                Q.push(b_node)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef long long _build_compressed_graph(
    long long[:] link_idx,
    long long[:] links_index,
    long long[:] link_edge,
    long long[:] a_nodes,
    long long[:] b_nodes,
    signed char[:] directions,
    long long link_id_max,
    long long[:] simplified_links,
    signed char[:] simplified_directions,
    long long[:] counts,
    long long[:] all_links,
    long long[:] compressed_dir,
    long long[:] compressed_a_node,
    long long[:] compressed_b_node
) noexcept nogil:
    cdef:
        long long slink = 0
        long long pre_link, n, first_node, lidx, a_node, b_node
        bint ab_dir, ba_dir
        long drc
        Py_ssize_t k

    # For each link we have marked for examining
    for pre_link in link_edge:
        if simplified_links[pre_link] >= 0:
            continue

        # We grab the initial information for that link
        lidx = link_idx[pre_link]
        a_node = a_nodes[lidx]
        b_node = b_nodes[lidx]
        drc = directions[lidx]

        n = a_node if counts[a_node] == 2 else b_node
        first_node = b_node if counts[a_node] == 2 else a_node

        # True if there exists a link in the direction ab (or ba), could be two links, or a single bidirectional link
        ab_dir = False if (first_node == a_node and drc < 0) or (first_node == b_node and drc > 0) else True
        ba_dir = False if (first_node == a_node and drc > 0) or (first_node == b_node and drc < 0) else True

        # While the node we are looking at, n, has degree two, we can continue to compress
        while counts[n] == 2:
            simplified_links[pre_link] = slink  # Mark link for removal
            simplified_directions[pre_link] = -1 if a_node == n else 1

            # Gets the link from the list that is not the link we are coming from
            for k in range(links_index[n], links_index[n + 1]):
                if pre_link != all_links[k]:
                    pre_link = all_links[k]
                    break

            lidx = link_idx[pre_link]
            a_node = a_nodes[lidx]
            b_node = b_nodes[lidx]
            drc = directions[lidx]
            ab_dir = False if (n == a_node and drc < 0) or (n == b_node and drc > 0) else ab_dir
            ba_dir = False if (n == a_node and drc > 0) or (n == b_node and drc < 0) else ba_dir
            n = (
                a_nodes[lidx]
                if n == b_nodes[lidx]
                else b_nodes[lidx]
            )

        simplified_links[pre_link] = slink
        simplified_directions[pre_link] = -1 if a_node == n else 1
        last_node = b_node if counts[a_node] == 2 else a_node
        # major_nodes[slink] = [first_node, last_node]

        # Available directions are NOT indexed like the other arrays
        compressed_a_node[slink] = first_node
        compressed_b_node[slink] = last_node
        if ab_dir:
            if ba_dir:
                compressed_dir[slink] = 0
            else:
                compressed_dir[slink] = 1
        elif ba_dir:
            compressed_dir[slink] = -1
        else:
            compressed_dir[slink] = -999
        slink += 1

    return slink


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void _back_fill(long long[:] links_index, long long max_node) noexcept:
    cdef Py_ssize_t i

    for i in range(max_node + 1, 0, -1):
        links_index[i - 1] = links_index[i] if links_index[i - 1] == -1 else links_index[i - 1]


def build_compressed_graph(graph, remove_dead_ends=True):
    # General notes:
    # Anything that uses graph.network is operating on the **mixed** graph. This graph has both directed and undirected
    # edges. Anything that uses graph.graph is operating on the **directed** graph. This graph has only directed edges,
    # they may be backwards but they are directed

    directed_node_max = max(graph.graph.a_node.values.max(), graph.graph.b_node.values.max())
    in_degree = np.bincount(graph.graph.b_node.values, minlength=directed_node_max + 1)
    out_degree = np.bincount(graph.graph.a_node.values, minlength=directed_node_max + 1)

    centroid_idx = graph.nodes_to_indices[graph.centroids]
    in_degree[centroid_idx] = -1
    out_degree[centroid_idx] = -1
    del centroid_idx

    df = pd.DataFrame(graph.network, copy=True)
    if remove_dead_ends:
        burnt_links = np.full(len(graph.graph), False, dtype=bool)
        _remove_dead_ends(
            graph.fs,
            graph.all_nodes,
            graph.nodes_to_indices,
            graph.graph.a_node.values,
            graph.graph.b_node.values,
            graph.graph.direction.values,
            in_degree,
            out_degree,
            burnt_links,
        )
        # Perhaps filter to unique link_ids? There'll be duplicates in here
        graph.dead_end_links = graph.graph.link_id.values[burnt_links]

        if graph.dead_end_links.shape[0]:
            df = df[~df.link_id.isin(graph.dead_end_links)]
    else:
        graph.dead_end_links = np.array([], dtype=np.int64)
    # Build link index
    link_id_max = df.link_id.max()

    link_idx = np.empty(link_id_max + 1, dtype=np.int64)
    link_idx[df.link_id] = np.arange(df.shape[0])

    nodes = np.hstack([df.a_node.values, df.b_node.values])
    links = np.hstack([df.link_id.values, df.link_id.values])
    # index (node) i has frequency counts[i]. This is just the number of edges that connect to a given node
    counts = np.bincount(nodes)

    idx = np.argsort(nodes)
    all_nodes = nodes[idx]
    all_links = links[idx]
    all_nodes_max = all_nodes.max()

    links_index = np.full(all_nodes_max + 2, -1, dtype=np.int64)
    nlist = np.arange(all_nodes_max + 2)

    y, x, _ = np.intersect1d(all_nodes, nlist, assume_unique=False, return_indices=True)
    links_index[y] = x[:]
    links_index[-1] = all_links.shape[0]

    _back_fill(links_index[:], all_nodes_max)

    # We keep all centroids for sure
    counts[graph.centroids] = 999

    degree_two = (counts == 2).astype(np.uint8)
    # Reorder and sum the degree two nodes by how they appear in the network, finds how a particular node is connected,
    # resulting values are
    # 0: This node is not of degree one. We're not interested in this case.
    # 1: This node is of degree one AND, has either incoming or outgoing flow. We can remove these these.
    link_edge = df.link_id.values[
        degree_two[df.a_node.values] + degree_two[df.b_node.values] == 1
    ].astype(np.int64)

    simplified_links = np.full(link_id_max + 1, -1, dtype=np.int64)
    simplified_directions = np.zeros(link_id_max + 1, dtype=np.int8)

    compressed_dir = np.zeros(link_id_max + 1, dtype=np.int64)
    compressed_a_node = np.zeros(link_id_max + 1, dtype=np.int64)
    compressed_b_node = np.zeros(link_id_max + 1, dtype=np.int64)

    slink = _build_compressed_graph(
        link_idx[:],
        links_index[:],
        link_edge[:],
        df.a_node.values[:],
        df.b_node.values[:],
        df.direction.values[:],
        link_id_max,
        simplified_links[:],
        simplified_directions[:],
        counts[:],
        all_links[:],
        compressed_dir[:],
        compressed_a_node[:],
        compressed_b_node[:]
    )

    links_to_remove = (simplified_links >= 0).nonzero()[0]
    if links_to_remove.shape[0]:
        df = df[~df.link_id.isin(links_to_remove)]
        df = df[df.a_node != df.b_node]

    comp_lnk = pd.DataFrame(
        {
            "a_node": compressed_a_node[:slink],
            "b_node": compressed_b_node[:slink],
            "direction": compressed_dir[:slink],
            "link_id": np.arange(slink),
        }
    )

    # Link compression can introduce new simple cycles into the graph
    comp_lnk = comp_lnk[comp_lnk.a_node != comp_lnk.b_node]

    max_link_id = link_id_max * 10
    comp_lnk.link_id += max_link_id

    df = pd.concat([df, comp_lnk])
    df = df[["id", "link_id", "a_node", "b_node", "direction"]]
    properties = graph._build_directed_graph(df, graph.centroids)
    graph.compact_all_nodes = properties[0]
    graph.compact_num_nodes = properties[1]
    graph.compact_nodes_to_indices = properties[2]
    graph.compact_fs = properties[3]
    graph.compact_graph = properties[4]

    crosswalk = pd.DataFrame(
        {
            "link_id": np.arange(simplified_directions.shape[0]),
            "link_direction": simplified_directions,
            "compressed_link": simplified_links,
            "compressed_direction": np.ones(simplified_directions.shape[0], dtype=np.int8),
        }
    )

    crosswalk = crosswalk[crosswalk.compressed_link >= 0]
    crosswalk.compressed_link += max_link_id

    cw2 = pd.DataFrame(crosswalk, copy=True)
    cw2.link_direction *= -1
    cw2.compressed_direction = -1

    crosswalk = pd.concat([crosswalk, cw2])
    crosswalk = crosswalk.assign(key=crosswalk.compressed_link * crosswalk.compressed_direction)
    crosswalk.drop(["compressed_link", "compressed_direction"], axis=1, inplace=True)

    final_ids = pd.DataFrame(graph.compact_graph[["id", "link_id", "direction"]], copy=True)
    final_ids = final_ids.assign(key=final_ids.link_id * final_ids.direction)
    final_ids.drop(["link_id", "direction"], axis=1, inplace=True)

    agg_crosswalk = crosswalk.merge(final_ids, on="key")
    agg_crosswalk.loc[:, "key"] = agg_crosswalk.link_id * agg_crosswalk.link_direction
    agg_crosswalk.drop(["link_id", "link_direction"], axis=1, inplace=True)

    direct_crosswalk = final_ids[final_ids.key.abs() < max_link_id]

    crosswalk = pd.concat([agg_crosswalk, direct_crosswalk])[["key", "id"]]
    crosswalk.columns = ["__graph_correlation_key__", "__compressed_id__"]

    graph.graph = graph.graph.assign(__graph_correlation_key__=graph.graph.link_id * graph.graph.direction)
    graph.graph = graph.graph.merge(crosswalk, on="__graph_correlation_key__", how="left")
    graph.graph.drop(["__graph_correlation_key__"], axis=1, inplace=True)

    # If will refer all the links that have no correlation to an element beyond the last link
    # This element will always be zero during assignment
    graph.graph.__compressed_id__ = graph.graph.__compressed_id__.fillna(
        graph.compact_graph.id.max() + 1
    ).astype(np.int64)


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
def create_compressed_link_network_mapping(graph):
    # Cache the result, this isn't a huge computation but isn't worth doing twice
    if (
        graph.compressed_link_network_mapping_idx is not None
        and graph.compressed_link_network_mapping_data is not None
        and graph.network_compressed_node_mapping is not None
    ):
        return (
            graph.compressed_link_network_mapping_idx,
            graph.compressed_link_network_mapping_data,
            graph.network_compressed_node_mapping,
        )

    cdef:
        long long i, j, a_node, x, b_node, tmp, compressed_id, non_duplicated_idx
        long long[:] b
        long long[:] values
        signed char[:] directions
        np.uint32_t[:] idx
        np.int64_t[::] data
        np.int32_t[:] node_mapping
        np.int64_t[:, :] non_duplicated
        signed char direction

    # This method requires that graph.graph is sorted on the a_node IDs, since that's done already we don't
    # bother redoing sorting it.

    # Some links are completely removed from the network, they are assigned ID `graph.compact_graph.id.max() + 1`,
    # we skip them.
    filtered = graph.graph[graph.graph.__compressed_id__ != graph.compact_graph.id.max() + 1]
    filtered = filtered[["__compressed_id__", "a_node", "b_node", "link_id", "direction"]]
    duplicated = filtered.__compressed_id__.duplicated(keep=False)
    gb = filtered[duplicated].groupby(by="__compressed_id__", sort=True)

    idx = np.zeros(graph.compact_num_links + 1, dtype=np.uint32)
    data = np.zeros(len(filtered), dtype=np.int64)
    node_mapping = np.full(graph.num_nodes, -1, dtype=np.int32)

    compact_a_nodes = graph.compact_graph["a_node"].to_numpy()
    compact_b_nodes = graph.compact_graph["b_node"].to_numpy()

    non_duplicated_idx = 0
    non_duplicated = filtered[~duplicated].sort_values(by="__compressed_id__").to_numpy()

    i = 0
    # This should be possible to parallelise, each thread gets a segment of the bincount below, they compute their
    # respective idx and data, then the end i value from the first segment is added to the idx of the segment after and
    # so on. Then the idx and data values are concatenated
    for compressed_id, count in enumerate(np.bincount(filtered["__compressed_id__"].to_numpy())):
        # We separate the easy un-compressible link path from the compressible link path
        # gb.get_group and those sorted searches are rather slow
        if count == 1:
            compressed_id, a_node, b_node, link_id, direction = non_duplicated[non_duplicated_idx]
            non_duplicated_idx += 1
            idx[compressed_id] = i
            data[i] = link_id * direction
            node_mapping[a_node] = compact_a_nodes[compressed_id]
            node_mapping[b_node] = compact_b_nodes[compressed_id]
        else:
            df = gb.get_group(compressed_id)
            idx[compressed_id] = i
            values = df.link_id.to_numpy()
            directions = df.direction.to_numpy()
            a = df.a_node.to_numpy()
            b = df.b_node.to_numpy()

            # In order to ensure that the link IDs come out in the correct order we must walk the links
            # we do this assuming the `a` array is sorted.
            j = 0

            # Find the missing a_node, this is the starting of the chain. We cannot rely on the node ordering to do a
            # simple lookup
            a_node = x = a[np.isin(a, b, invert=True, assume_unique=True)][0]
            while True:
                tmp = a.searchsorted(x)
                if tmp < len(a) and a[tmp] == x:
                    x = b[tmp]
                    data[i + j] = values[tmp] * directions[tmp]
                else:
                    break
                j += 1

            b_node = x
            node_mapping[a_node] = compact_a_nodes[compressed_id]
            node_mapping[b_node] = compact_b_nodes[compressed_id]
        i += count

    idx[-1] = i

    graph.compressed_link_network_mapping_idx = np.array(idx)
    graph.compressed_link_network_mapping_data = np.array(data)
    graph.network_compressed_node_mapping = np.array(node_mapping)

    return (
        graph.compressed_link_network_mapping_idx,
        graph.compressed_link_network_mapping_data,
        graph.network_compressed_node_mapping
    )
