from aequilibrae.utils.find_non_applicable_turn_restrictions import find_non_applicable_turn_restrictions


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


def test_find_non_applicable_turn_restrictions(sioux_falls_example):
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
            (from_node, via_node, to_node, None, "c"),
        )
        restriction_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            """
            UPDATE links
            SET modes = 'x'
            WHERE a_node = ? AND b_node = ? AND direction >= 0
            """,
            (from_node, via_node),
        )

    df = find_non_applicable_turn_restrictions(sioux_falls_example)
    assert not df.empty

    row = df.loc[df.restriction_id == restriction_id].iloc[0]
    assert row.turn_type == "ban"
    assert "no_mode_overlap_from_leg" in row.issue
