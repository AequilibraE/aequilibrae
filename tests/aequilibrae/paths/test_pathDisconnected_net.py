import os
from tempfile import gettempdir
from uuid import uuid4
from os.path import join
from itertools import product

import pytest

from aequilibrae.paths import Graph
from aequilibrae.paths.results import PathResults
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def project_fixture():
    project = create_example(join(gettempdir(), "test_path_disconnected" + uuid4().hex))
    yield project
    project.close()


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_path_disconnected_delete_link(project_fixture, early_exit, a_star):
    project = project_fixture
    with project.db_connection as conn:
        conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])

    project.network.build_graphs()
    g = project.network.graphs["c"]
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    r = PathResults()
    r.prepare(g)
    r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
    assert r.path is None, "Failed to return None for disconnected"
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"


@pytest.mark.parametrize("early_exit,a_star", product([True, False], repeat=2))
def test_path_disconnected_penalize_link_in_memory(project_fixture, early_exit, a_star):
    project = project_fixture
    links = [2, 4, 5, 14]

    project.network.build_graphs()
    g = project.network.graphs["c"]
    g.exclude_links(links)
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    r = PathResults()
    r.prepare(g)
    r.compute_path(1, 5, early_exit=early_exit, a_star=a_star)
    assert r.path is None, "Failed to return None for disconnected"
    r.compute_path(1, 2, early_exit=early_exit, a_star=a_star)
    assert len(r.path) == 1, "Returned the wrong thing for existing path on disconnected network"
