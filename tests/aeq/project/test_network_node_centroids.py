def test_no_centroids(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.execute("Update Nodes set is_centroid=0")

    sioux_falls_example.network.build_graphs(modes=["c"])
    sioux_falls_example.network.build_graphs()
