from itertools import product

import pytest

from aequilibrae.paths.results import PathResults


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_path_disconnected_delete_link(sioux_falls_example, early_exit, a_star):
    with sioux_falls_example.db_connection as conn:
        conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])

    sioux_falls_example.network.build_graphs()
    g = sioux_falls_example.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    r = PathResults()
    r.prepare(g)
    r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
    assert r.path is None, "Failed to return None for disconnected"
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_path_disconnected_penalize_link_in_memory(sioux_falls_example, early_exit, a_star):
    links = [2, 4, 5, 14]

    sioux_falls_example.network.build_graphs()
    g = sioux_falls_example.network.graphs["c"]
    g.exclude_links(links)
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    r = PathResults()
    r.prepare(g)
    r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
    assert r.path is None, "Failed to return None for disconnected"
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"
