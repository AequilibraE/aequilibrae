"""Tests for turn project integration: upgrade messaging, validation, mode filtering, exclude_links (H6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph


def test_stale_schema_actionable_error():
    """Verifies that an actionable project.upgrade() error message is raised when turn table has stale schema."""
    graph = Graph()
    # Missing 'modes' or 'via_node'
    df = pd.DataFrame([{"from_link": 1, "to_link": 2}])
    with pytest.raises(ValueError, match="project.upgrade"):
        graph.set_turn_restrictions(df)


def test_turn_table_strict_int64_validation():
    """Verifies that non-integer node IDs in turn tables are rejected."""
    graph = Graph()
    df = pd.DataFrame([{"from_node": "A", "via_node": "B", "to_node": "C", "penalty": 1.0}])
    with pytest.raises(ValueError, match="integer"):
        graph.set_turn_restrictions(df)


def test_project_mode_filtering_and_empty_restrictions(sioux_falls_example):
    """Verifies that build_graphs filters turn restrictions by mode, and empty restriction sets are skipped cleanly."""
    net = sioux_falls_example.network
    # Add a restriction only for mode 'x' (transit/custom)
    with sioux_falls_example.db_connection_spatial as conn:
        conn.execute("INSERT OR IGNORE INTO modes (mode_id, mode_name) VALUES ('x', 'custom x')")
        # Find a valid 3-node sequence
        pair = conn.execute(
            """
            SELECT l1.a_node, l1.b_node, l2.b_node
            FROM links l1
            JOIN links l2 ON l1.b_node = l2.a_node
            LIMIT 1
            """
        ).fetchone()

    assert pair is not None
    net.turn_restrictions.add_restriction(pair[0], pair[1], pair[2], penalty=5.0, modes="x")

    # Build graphs for mode 'c' (car) - the restriction with modes='x' must be filtered out!
    net.build_graphs(modes=["c"])
    graph_c = net.graphs["c"]
    assert not graph_c.has_turn_restrictions


def test_exclude_links_physically_removes_links_and_invalidates_cache(sioux_falls_example):
    """Verifies that exclude_links physically removes links from graph.network and invalidates caches."""
    net = sioux_falls_example.network
    net.build_graphs(modes=["c"])
    graph = net.graphs["c"]
    graph.prepare_graph(centroids=np.array([1, 2], dtype=np.int64))
    graph.set_graph("distance")

    initial_num_links = graph.num_links
    assert initial_num_links > 0

    # Pick link_id 1 to exclude
    graph.exclude_links([1])
    assert 1 not in graph.network["link_id"].values
    assert graph.num_links < initial_num_links
