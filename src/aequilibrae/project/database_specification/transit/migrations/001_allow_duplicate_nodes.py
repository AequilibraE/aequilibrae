import sqlite3
from typing import Optional


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    # The transit graph can create a lot of nodes on top of each other, these shouldn't
    # be cannibalised, they also shouldn't have to be scattered to fit in the database.
    # So we remove the related triggers on the transit database.
    transit_conn.execute("DROP TRIGGER IF EXISTS no_duplicate_node")
    transit_conn.execute("DROP TRIGGER IF EXISTS cannibalize_node_abort_when_centroid")
    transit_conn.execute("DROP TRIGGER IF EXISTS cannibalize_node")
