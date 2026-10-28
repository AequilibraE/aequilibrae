import numpy as np


def test_connectivity_analysis(sioux_falls_example):
    network = sioux_falls_example.network
    graph = network.build_graph(mode="c", cost="distance", block_centroid_flows=False)

    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 0

    source = network.build()
    costs = source.field_values("distance").copy()
    costs[np.isin(source.link_ids, [5, 12, 19])] = np.inf  # Disconnect nodes 1, 2, and 6 from the rest of the graph
    graph.set_costs(costs)
    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 3
