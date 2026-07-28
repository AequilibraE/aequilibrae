import sqlite3

import pytest


def _sample_turn_pair(conn):
    return conn.execute(
        """
                                SELECT l1.a_node, l1.b_node, l2.b_node
        FROM links l1
        JOIN links l2 ON l1.b_node = l2.a_node
        WHERE INSTR(l1.modes, 'c') > 0
          AND INSTR(l2.modes, 'c') > 0
          AND l1.direction >= 0
          AND l2.direction >= 0
        LIMIT 1
        """
    ).fetchone()


def _sample_disconnected_pair(conn):
    return conn.execute(
        """
        SELECT n1.node_id, n1.node_id, n2.node_id
                FROM nodes n1
        JOIN nodes n2 ON n2.node_id != n1.node_id
        LIMIT 1
        """
    ).fetchone()


def test_turn_restrictions_has_modes_column(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(turn_restrictions)").fetchall()]
        assert "modes" in cols
        assert "from_node" in cols
        assert "via_node" in cols
        assert "to_node" in cols
        assert "geometry" in cols


def test_turn_restriction_no_duplicate_mode_overlap(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, None, "c"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (from_node, via_node, to_node, 12.0, "c"),
            )


def test_turn_restriction_no_duplicate_mode_overlap_different_mode_config(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute("INSERT OR IGNORE INTO modes (mode_id, mode_name) VALUES ('x', 'test mode x')")

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, 8.0, "cx"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (from_node, via_node, to_node, 5.0, "xc"),
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (from_node, via_node, to_node, None, "c"),
            )


def test_turn_restrictions_node_delete_blocked_with_links(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, 10.0, "c"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM nodes WHERE node_id = ?", (via_node,))


def test_mode_update_blocked_when_used_in_turn_restrictions(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, None, "c"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE modes SET mode_id='x' WHERE mode_id='c'")


def test_mode_delete_blocked_when_used_in_turn_restrictions(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, None, "c"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM modes WHERE mode_id='c'")


def test_turn_restriction_via_node_is_stored(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        conn.execute(
            """
            INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_node, via_node, to_node, None, "c"),
        )

        stored_via_node = conn.execute(
            """
            SELECT via_node
            FROM turn_restrictions
            WHERE from_node = ? AND via_node = ? AND to_node = ? AND modes = 'c'
            """,
            (from_node, via_node, to_node),
        ).fetchone()[0]

        assert stored_via_node == via_node


def test_turn_restriction_api_stores_infinite_penalty_as_prohibition(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

    restriction_id = sioux_falls_example.network.turn_restrictions.add_restriction(
        from_node, via_node, to_node, penalty=float("inf"), modes="c"
    )

    with sioux_falls_example.db_connection_spatial as conn:
        stored_penalty = conn.execute(
            "SELECT penalty FROM turn_restrictions WHERE restriction_id = ?",
            (restriction_id,),
        ).fetchone()[0]

    assert stored_penalty is None


def test_turn_restriction_api_rejects_negative_penalty(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

    with pytest.raises(ValueError, match="Negative turn penalties"):
        sioux_falls_example.network.turn_restrictions.add_restriction(
            from_node, via_node, to_node, penalty=-1.0, modes="c"
        )


def test_turn_restriction_api_update_can_set_prohibition(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_turn_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

    turns = sioux_falls_example.network.turn_restrictions
    restriction_id = turns.add_restriction(from_node, via_node, to_node, penalty=8.0, modes="c")

    assert turns.update_restriction(restriction_id, penalty=None)

    with sioux_falls_example.db_connection_spatial as conn:
        stored_penalty = conn.execute(
            "SELECT penalty FROM turn_restrictions WHERE restriction_id = ?",
            (restriction_id,),
        ).fetchone()[0]

    assert stored_penalty is None


def test_turn_restriction_requires_node_consistency(sioux_falls_example):
    with sioux_falls_example.db_connection_spatial as conn:
        pair = _sample_disconnected_pair(conn)
        assert pair is not None
        from_node, via_node, to_node = pair

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (from_node, via_node, to_node, None, "c"),
            )
