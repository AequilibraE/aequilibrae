def test_connectivity_analysis(sioux_falls_example):
    network = sioux_falls_example.network
    network.build_graphs()
    graph = network.graphs["c"]

    graph.block_centroid_flows = False

    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 0

    graph.exclude_links([5, 12, 19])  # Disconnect nodes 1, 2, and 6 from the reset of the graph
    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 3
