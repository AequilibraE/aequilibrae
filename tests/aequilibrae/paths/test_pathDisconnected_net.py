from itertools import product

from aequilibrae.paths.results import PathResults


def test_path_disconnected_delete_link(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])

    sioux_falls_example.network.build_graphs()
    graph = sioux_falls_example.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)

    for early_exit, a_star in product([True, False], repeat=2):
        result = PathResults(graph, 1, 5, early_exit=early_exit, a_star=a_star)
        assert result.path is None, "Failed to return None for disconnected"
        result.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
        assert len(result.path) == 1, "Returned the wrong thing for existing path on disconnected network"


def test_path_disconnected_penalize_link_in_memory(sioux_falls_example):
    links = [2, 4, 5, 14]

    sioux_falls_example.network.build_graphs()
    graph = sioux_falls_example.network.graphs["c"]
    graph.exclude_links(links)
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)

    for early_exit, a_star in product([True, False], repeat=2):
        result = PathResults(graph, 1, 5, early_exit=early_exit, a_star=a_star)
        assert result.path is None, "Failed to return None for disconnected"
        result.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
        assert len(result.path) == 1, "Returned the wrong thing for existing path on disconnected network"
