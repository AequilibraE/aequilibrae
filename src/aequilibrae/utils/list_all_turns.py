import numpy as np
import pandas as pd

from aequilibrae.project.project import Project


def _build_directed_edges(links, nodes, mode: str):
    links = links.loc[
        links["modes"].astype(str).str.lower().str.contains(mode.lower(), regex=False),
        ["link_id", "a_node", "b_node", "direction"],
    ].copy()

    if links.empty:
        return pd.DataFrame({"from_node": [], "to_node": [], "link_id": [], "direction": [], "heading": []})

    node_xy = nodes[["node_id", "longitude", "latitude"]].copy()
    node_xy = node_xy.dropna(subset=["node_id", "longitude", "latitude"])

    node_id = node_xy["node_id"].to_numpy(dtype=np.int64)
    node_x = node_xy["longitude"].to_numpy(dtype=np.float64)
    node_y = node_xy["latitude"].to_numpy(dtype=np.float64)

    order = np.argsort(node_id)
    sorted_ids = node_id[order]

    links_a = links["a_node"].to_numpy(dtype=np.int64)
    links_b = links["b_node"].to_numpy(dtype=np.int64)

    idx_a = np.searchsorted(sorted_ids, links_a)
    idx_b = np.searchsorted(sorted_ids, links_b)

    valid_a = (idx_a < sorted_ids.size) & (sorted_ids[idx_a] == links_a)
    valid_b = (idx_b < sorted_ids.size) & (sorted_ids[idx_b] == links_b)
    valid = valid_a & valid_b

    links = links.iloc[valid].copy()
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]

    if links.empty:
        return pd.DataFrame({"from_node": [], "to_node": [], "link_id": [], "direction": [], "heading": []})

    sorted_x = node_x[order]
    sorted_y = node_y[order]

    ax = sorted_x[idx_a]
    ay = sorted_y[idx_a]
    bx = sorted_x[idx_b]
    by = sorted_y[idx_b]

    link_id = links["link_id"].to_numpy(dtype=np.int64)
    a_node = links["a_node"].to_numpy(dtype=np.int64)
    b_node = links["b_node"].to_numpy(dtype=np.int64)
    direction = links["direction"].to_numpy(dtype=np.int8)

    allow_ab = direction >= 0
    allow_ba = direction <= 0

    heading_ab = (np.degrees(np.arctan2(by - ay, bx - ax)) + 360.0) % 360.0
    heading_ba = (np.degrees(np.arctan2(ay - by, ax - bx)) + 360.0) % 360.0

    from_node = np.concatenate([a_node[allow_ab], b_node[allow_ba]])
    to_node = np.concatenate([b_node[allow_ab], a_node[allow_ba]])
    lids = np.concatenate([link_id[allow_ab], link_id[allow_ba]])
    dirs = np.concatenate(
        [
            np.ones(np.sum(allow_ab), dtype=np.int8),
            -np.ones(np.sum(allow_ba), dtype=np.int8),
        ]
    )
    headings = np.concatenate([heading_ab[allow_ab], heading_ba[allow_ba]])
    return pd.DataFrame(
        {
            "from_node": from_node,
            "to_node": to_node,
            "link_id": lids,
            "direction": dirs,
            "heading": headings,
        }
    )


def _index_by_node(node_array: np.ndarray):
    order = np.argsort(node_array)
    sorted_nodes = node_array[order]
    unique_nodes, starts, counts = np.unique(sorted_nodes, return_index=True, return_counts=True)
    return order, unique_nodes, starts, counts


def _classify_turns(turn_angle: np.ndarray) -> np.ndarray:
    deg = np.asarray(turn_angle, dtype=np.float64)
    return np.select([np.abs(deg) >= 150.0, deg > 30.0, deg < -30.0], ["u_turn", "left", "right"], default="through")


def list_left_turns(
    project: Project,
    mode: str = "c",
):
    links = project.network.links.data[["link_id", "a_node", "b_node", "direction", "modes"]].copy()
    nodes = project.network.nodes.data[["node_id", "longitude", "latitude"]].copy()

    edges = _build_directed_edges(links, nodes, mode)

    columns = ["node_id", "from_link_id", "from_dir", "to_link_id", "to_dir", "turn_angle", "turn_type"]

    if edges.empty:
        return pd.DataFrame(columns=columns)

    in_order, in_nodes, in_starts, in_counts = _index_by_node(edges["to_node"].to_numpy(dtype=np.int64))
    out_order, out_nodes, out_starts, out_counts = _index_by_node(edges["from_node"].to_numpy(dtype=np.int64))

    common_nodes, in_idx, out_idx = np.intersect1d(in_nodes, out_nodes, return_indices=True)
    if common_nodes.size == 0:
        return pd.DataFrame(columns=columns)

    edge_link = edges["link_id"].to_numpy(dtype=np.int64)
    edge_dir = edges["direction"].to_numpy(dtype=np.int8)
    edge_heading = edges["heading"].to_numpy(dtype=np.float64)

    node_chunks, from_link_chunks, from_dir_chunks, to_link_chunks, to_dir_chunks, angle_chunks = [], [], [], [], [], []
    for k in range(common_nodes.size):
        in_pos = in_idx[k]
        out_pos = out_idx[k]

        in_slice = in_order[in_starts[in_pos] : in_starts[in_pos] + in_counts[in_pos]]
        out_slice = out_order[out_starts[out_pos] : out_starts[out_pos] + out_counts[out_pos]]

        if in_slice.size == 0 or out_slice.size == 0:
            continue

        incoming_heading = edge_heading[in_slice]
        outgoing_heading = edge_heading[out_slice]

        turn_angles = ((outgoing_heading[None, :] - incoming_heading[:, None] + 540.0) % 360.0) - 180.0

        row_idx, col_idx = np.indices(turn_angles.shape)
        row_idx = row_idx.ravel()
        col_idx = col_idx.ravel()

        in_mov = in_slice[row_idx]
        out_mov = out_slice[col_idx]

        node_chunks.append(np.full(row_idx.size, common_nodes[k], dtype=np.int64))
        from_link_chunks.append(edge_link[in_mov])
        from_dir_chunks.append(edge_dir[in_mov])
        to_link_chunks.append(edge_link[out_mov])
        to_dir_chunks.append(edge_dir[out_mov])
        angle_chunks.append(turn_angles[row_idx, col_idx])

    if not node_chunks:
        data = pd.DataFrame(columns=columns)
    else:
        all_angles = np.concatenate(angle_chunks)
        data = pd.DataFrame(
            {
                "node_id": np.concatenate(node_chunks),
                "from_link_id": np.concatenate(from_link_chunks),
                "from_dir": np.concatenate(from_dir_chunks),
                "to_link_id": np.concatenate(to_link_chunks),
                "to_dir": np.concatenate(to_dir_chunks),
                "turn_angle": all_angles,
                "turn_type": _classify_turns(all_angles),
            }
        )

    return data
