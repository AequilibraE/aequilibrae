from itertools import product

from aequilibrae.paths.results import PathResults


def test_path_disconnected_delete_link(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])

    sioux_falls_example.network.build_graphs()
    g = sioux_falls_example.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)

    for early_exit, a_star in product([True, False], repeat=2):
        r = PathResults()
        r.prepare(g)
        r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
        assert r.path is None, "Failed to return None for disconnected"
        r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
        assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"


def test_path_disconnected_penalize_link_in_memory(sioux_falls_example):
    links = [2, 4, 5, 14]

    sioux_falls_example.network.build_graphs()
    g = sioux_falls_example.network.graphs["c"]
    g.exclude_links(links)
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)

    for early_exit, a_star in product([True, False], repeat=2):
        r = PathResults()
        r.prepare(g)
        r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
        assert r.path is None, "Failed to return None for disconnected"
        r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
        assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"
