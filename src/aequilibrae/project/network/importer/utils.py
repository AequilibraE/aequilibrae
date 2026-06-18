import pandas as pd


def compute_node_modes(node_ids, links: pd.DataFrame, fallback: str = "") -> list:
    nodes_col = pd.concat([links["a_node"], links["b_node"]], ignore_index=True)
    modes_col = pd.concat([links["modes"], links["modes"]], ignore_index=True).map(set)
    per_node = (
        pd.DataFrame({"node": nodes_col, "modes": modes_col})
        .groupby("node")["modes"]
        .agg(lambda s: "".join(sorted(set().union(*s))))
        .to_dict()
    )
    return [per_node.get(int(nid), fallback) for nid in node_ids]
