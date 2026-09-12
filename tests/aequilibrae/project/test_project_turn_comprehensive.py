"""Comprehensive project integration tests covering finite, prohibited, U-turn, and multimode restrictions."""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import pytest


def _find_connected_turn_pair(conn, mode="c"):
    return conn.execute(
        f"""
        SELECT l1.a_node, l1.b_node, l2.b_node, l1.distance + l2.distance
        FROM links l1
        JOIN links l2 ON l1.b_node = l2.a_node
        WHERE INSTR(l1.modes, '{mode}') > 0
          AND INSTR(l2.modes, '{mode}') > 0
          AND l1.direction = 1
          AND l2.direction = 1
          AND l1.a_node != l2.b_node
        LIMIT 1
        """
    ).fetchone()


def test_project_finite_turn_penalty(sioux_falls_example):
    """Verifies that an active finite turn penalty in a project graph increases path cost correctly."""
    with sioux_falls_example.db_connection as conn:
        turn = _find_connected_turn_pair(conn, "c")
        assert turn is not None
        u, v, w, _ = turn
        conn.execute(
            "INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes) VALUES (?, ?, ?, 15.0, 'c')",
            (u, v, w),
        )

    sioux_falls_example.network.build_graphs(modes=["c"])
    g = sioux_falls_example.network.graphs["c"]
    assert g.has_turn_restrictions

    g.prepare_graph(centroids=np.array([u, w], dtype=np.int64), remove_dead_ends=False)
    g.set_graph("distance")

    path_res = g.compute_path(u, w)
    assert path_res.path is not None

    # Compare with graph having no turns
    g_noturns = sioux_falls_example.network.graphs["c"]
    g_noturns.clear_turn_restrictions()
    g_noturns.prepare_graph(centroids=np.array([u, w], dtype=np.int64), remove_dead_ends=False)
    g_noturns.set_graph("distance")
    path_noturns = g_noturns.compute_path(u, w)

    # The finite turn penalty is in the same units as cost, so cost should be exactly 15.0 higher
    assert path_res.milepost[-1] == pytest.approx(path_noturns.milepost[-1] + 15.0)


def test_project_prohibited_turn_restriction(sioux_falls_example):
    """Verifies that an active turn prohibition in a project graph blocks the direct turn."""
    with sioux_falls_example.db_connection as conn:
        turn = _find_connected_turn_pair(conn, "c")
        assert turn is not None
        u, v, w, _ = turn
        conn.execute(
            "INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes) VALUES (?, ?, ?, NULL, 'c')",
            (u, v, w),
        )

    sioux_falls_example.network.build_graphs(modes=["c"])
    g = sioux_falls_example.network.graphs["c"]
    assert g.has_turn_restrictions

    g.prepare_graph(centroids=np.array([u, w], dtype=np.int64), remove_dead_ends=False)
    g.set_graph("distance")

    path_res = g.compute_path(u, w)
    if path_res.path is not None:
        # If an alternative detour path exists, verify it does NOT use the prohibited sequence u -> v -> w
        nodes = [int(n) for n in path_res.path_nodes]
        for i in range(len(nodes) - 2):
            assert (nodes[i], nodes[i + 1], nodes[i + 2]) != (u, v, w)


def test_project_uturn_policy_from_about_table(sioux_falls_example):
    """Verifies that allow_uturns configured in the project about table correctly controls allow_path_uturns."""
    # 1. Enable U-turns in project about table
    with sioux_falls_example.db_connection as conn:
        conn.execute("INSERT OR REPLACE INTO about (infoname, infovalue) VALUES ('allow_uturns', '1')")

    sioux_falls_example.network.build_graphs(modes=["c"])
    g_enabled = sioux_falls_example.network.graphs["c"]
    assert g_enabled._allow_path_uturns is True

    # 2. Disable U-turns in project about table
    with sioux_falls_example.db_connection as conn:
        conn.execute("INSERT OR REPLACE INTO about (infoname, infovalue) VALUES ('allow_uturns', '0')")

    sioux_falls_example.network.build_graphs(modes=["c"])
    g_disabled = sioux_falls_example.network.graphs["c"]
    assert g_disabled._allow_path_uturns is False


def test_project_multimode_turn_restrictions(sioux_falls_example):
    """Verifies that project build_graphs correctly isolates single-mode and multi-mode turn restrictions."""
    with sioux_falls_example.db_connection as conn:
        # Ensure mode 'b' (bus) exists
        conn.execute("INSERT OR IGNORE INTO modes (mode_id, mode_name) VALUES ('b', 'bus')")

        # Give all links mode 'b' in addition to 'c' so both modes have an identical network
        conn.execute("UPDATE links SET modes = 'cb'")

        # Find two valid 3-node sequences
        turns = conn.execute(
            """
            SELECT l1.a_node, l1.b_node, l2.b_node
            FROM links l1
            JOIN links l2 ON l1.b_node = l2.a_node
            WHERE l1.direction = 1 AND l2.direction = 1 AND l1.a_node != l2.b_node
            LIMIT 2
            """
        ).fetchall()
        assert len(turns) >= 2
        t1, t2 = turns[0], turns[1]

        # t1: mode 'c' only (finite penalty 20.0)
        conn.execute(
            "INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes) VALUES (?, ?, ?, 20.0, 'c')",
            t1,
        )
        # t2: mode 'b' only (finite penalty 40.0)
        conn.execute(
            "INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes) VALUES (?, ?, ?, 40.0, 'b')",
            t2,
        )

    # Build graphs for both modes
    sioux_falls_example.network.build_graphs(modes=["c", "b"])
    graph_c = sioux_falls_example.network.graphs["c"]
    graph_b = sioux_falls_example.network.graphs["b"]

    assert graph_c.has_turn_restrictions
    assert graph_b.has_turn_restrictions

    # graph_c should only have t1, not t2
    c_turns = set(zip(graph_c._turn_restrictions.from_node, graph_c._turn_restrictions.via_node, graph_c._turn_restrictions.to_node))
    assert t1 in c_turns
    assert t2 not in c_turns

    # graph_b should only have t2, not t1
    b_turns = set(zip(graph_b._turn_restrictions.from_node, graph_b._turn_restrictions.via_node, graph_b._turn_restrictions.to_node))
    assert t2 in b_turns
    assert t1 not in b_turns

    # Verify path costs for mode 'c' vs mode 'b' on movement t1
    u1, v1, w1 = t1
    graph_c.prepare_graph(centroids=np.array([u1, w1], dtype=np.int64), remove_dead_ends=False)
    graph_c.set_graph("distance")
    cost_c = graph_c.compute_path(u1, w1).milepost[-1]

    graph_b.prepare_graph(centroids=np.array([u1, w1], dtype=np.int64), remove_dead_ends=False)
    graph_b.set_graph("distance")
    cost_b = graph_b.compute_path(u1, w1).milepost[-1]

    # Mode 'c' should pay the 20.0 penalty, while mode 'b' should not pay it
    assert cost_c == pytest.approx(cost_b + 20.0)
