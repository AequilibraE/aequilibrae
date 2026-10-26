def test_no_centroids(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.execute("Update Nodes set is_centroid=0")

    network = sioux_falls_example.network
    graph = network.build_graph(mode="c", cost="distance")
    assert graph.num_zones == 0
