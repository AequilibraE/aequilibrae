import pandas as pd
import numpy as np
cimport cython


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef long long _build_compressed_graph(long long[:] link_idx,
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
                                  long long[:] compressed_b_node) noexcept nogil:
    cdef:
        long long slink = 0
        long long pre_link, n, first_node, lnk, lidx, a_node, b_node
        long ab_dir = 1  # These could definitely be smaller since we only ever use them in conditionals
        long ba_dir = 1
        long drc
        Py_ssize_t i, k

    for i in range(link_edge.shape[0]):
        pre_link = link_edge[i]

        if simplified_links[pre_link] >= 0:
            continue
        ab_dir = 1
        ba_dir = 1
        lidx = link_idx[pre_link]
        a_node = a_nodes[lidx]
        b_node = b_nodes[lidx]
        drc = directions[lidx]
        n = a_node if counts[a_node] == 2 else b_node
        first_node = b_node if counts[a_node] == 2 else a_node

        ab_dir = 0 if (first_node == a_node and drc < 0) or (first_node == b_node and drc > 0) else ab_dir
        ba_dir = 0 if (first_node == a_node and drc > 0) or (first_node == b_node and drc < 0) else ba_dir

        while counts[n] == 2:
            # assert (simplified_links[pre_link] >= 0), "How the heck did this happen?"
            simplified_links[pre_link] = slink
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
            ab_dir = 0 if (n == a_node and drc < 0) or (n == b_node and drc > 0) else ab_dir
            ba_dir = 0 if (n == a_node and drc > 0) or (n == b_node and drc < 0) else ba_dir
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
        if ab_dir > 0:
            if ba_dir > 0:
                compressed_dir[slink] = 0
            else:
                compressed_dir[slink] = 1
        elif ba_dir > 0:
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


def build_compressed_graph(graph):
    # Build link index
    link_id_max = graph.network.link_id.max()

    link_idx = np.empty(link_id_max + 1, dtype=np.int64)
    link_idx[graph.network.link_id] = np.arange(graph.network.shape[0])

    nodes = np.hstack([graph.network.a_node.values, graph.network.b_node.values])
    links = np.hstack([graph.network.link_id.values, graph.network.link_id.values])
    counts = np.bincount(nodes) # index (node) i has frequency counts[i]. This is just the number of
    # edges that connection to a given node

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

    truth = (counts == 2).astype(np.uint8)
    link_edge = truth[graph.network.a_node.values] + truth[graph.network.b_node.values]
    link_edge = graph.network.link_id.values[link_edge == 1].astype(np.int64)

    simplified_links = np.full(link_id_max + 1, -1, dtype=np.int64)
    simplified_directions = np.zeros(link_id_max + 1, dtype=np.int8)

    compressed_dir = np.zeros(link_id_max + 1, dtype=np.int64)
    compressed_a_node = np.zeros(link_id_max + 1, dtype=np.int64)
    compressed_b_node = np.zeros(link_id_max + 1, dtype=np.int64)

    slink = _build_compressed_graph(
        link_idx[:],
        links_index[:],
        link_edge[:],
        graph.network.a_node.values[:],
        graph.network.b_node.values[:],
        graph.network.direction.values[:],
        link_id_max,
        simplified_links[:],
        simplified_directions[:],
        counts[:],
        all_links[:],
        compressed_dir[:],
        compressed_a_node[:],
        compressed_b_node[:]
    )

    links_to_remove = np.argwhere(simplified_links >= 0)
    df = pd.DataFrame(graph.network, copy=True)
    if links_to_remove.shape[0]:
        df = df[~df.link_id.isin(links_to_remove[:, 0])]
        df = df[df.a_node != df.b_node]

    comp_lnk = pd.DataFrame(
        {
            "a_node": compressed_a_node[:slink],
            "b_node": compressed_b_node[:slink],
            "direction": compressed_dir[:slink],
            "link_id": np.arange(slink),
        }
    )
    max_link_id = link_id_max * 10
    comp_lnk.link_id += max_link_id

    df = pd.concat([df, comp_lnk])
    df = df[["id", "link_id", "a_node", "b_node", "direction"]]
    properties = graph._Graph__build_directed_graph(df, graph.centroids)  # FIXME: Don't circumvent name mangling
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
    graph.graph.loc[graph.graph.__compressed_id__.isna(), "__compressed_id__"] = graph.compact_graph.id.max() + 1
