def test_no_centroids(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.execute("Update Nodes set is_centroid=0")

    network = sioux_falls_example.network.build(modes=["c"])
    assert network.centroids is None
    graph = network.build_graph(mode="c", cost="distance")
    assert graph.num_zones == 0
