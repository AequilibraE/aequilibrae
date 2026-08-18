import logging

logger = logging.getLogger(__name__)


def migrate(*, closure):
    """Align transit origin node IDs with TAZ IDs using migration-owned SQL."""
    transit = closure["transit"]
    if transit.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'").fetchone() is None:
        return
    columns = {row[1] for row in transit.execute("PRAGMA table_info(nodes)").fetchall()}
    if not {"node_id", "period_id", "node_type", "taz_id"} <= columns:
        return

    transit.execute("UPDATE nodes SET taz_id=NULL WHERE CAST(taz_id AS TEXT)='' OR taz_id IS NULL")
    periods = [row[0] for row in transit.execute("SELECT DISTINCT period_id FROM nodes").fetchall()]
    transit.execute("CREATE TEMP TABLE aeq_node_map (period_id INTEGER, old_id INTEGER, new_id INTEGER)")
    try:
        for period_id in periods:
            rows = transit.execute(
                "SELECT node_id, taz_id, node_type FROM nodes WHERE period_id=? ORDER BY node_id", (period_id,)
            ).fetchall()
            reserved = {
                int(taz_id)
                for _, taz_id, node_type in rows
                if taz_id is not None and int(taz_id) > 0 and node_type in ("origin", "od")
            }
            next_id = max(reserved, default=0) + 1
            mapping = []
            for old_id, taz_id, node_type in rows:
                if taz_id is not None and int(taz_id) > 0 and node_type in ("origin", "od"):
                    new_id = int(taz_id)
                else:
                    while next_id in reserved:
                        next_id += 1
                    new_id = next_id
                    next_id += 1
                mapping.append((period_id, old_id, new_id))
            transit.executemany("INSERT INTO aeq_node_map VALUES (?,?,?)", mapping)

        if transit.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='links'").fetchone():
            transit.execute(
                """UPDATE links SET
                a_node=(SELECT new_id FROM aeq_node_map m WHERE m.period_id=links.period_id AND m.old_id=links.a_node),
                b_node=(SELECT new_id FROM aeq_node_map m WHERE m.period_id=links.period_id AND m.old_id=links.b_node)
                WHERE EXISTS (SELECT 1 FROM aeq_node_map m WHERE m.period_id=links.period_id
                              AND (m.old_id=links.a_node OR m.old_id=links.b_node))"""
            )
        transit.execute(
            """UPDATE nodes SET node_id=(SELECT new_id FROM aeq_node_map m
               WHERE m.period_id=nodes.period_id AND m.old_id=nodes.node_id)"""
        )
    finally:
        transit.execute("DROP TABLE aeq_node_map")
    logger.info("Aligned transit TAZ and node identifiers")
