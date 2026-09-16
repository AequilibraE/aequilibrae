import sqlite3
from typing import Optional


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 008 requires a project_conn connection")

    new_columns = {
        "links": [("fixed_cost_ab", "NUMERIC"), ("fixed_cost_ba", "NUMERIC"), ("other_attributes", "TEXT")],
        "nodes": [("other_attributes", "TEXT")],
    }

    for table, columns in new_columns.items():
        existing = {name for _, name, *_ in project_conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column, column_type in columns:
            if column not in existing:
                project_conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    documentation = [
        ("links", "fixed_cost_*", "Directional fixed costs (if any). Tolls, for example"),
        ("links", "other_attributes", "Other attributes of the link. Preferably in json format"),
        ("nodes", "other_attributes", "Other attributes of the node. Preferably in json format"),
    ]

    for name_table, attribute, description in documentation:
        documented = project_conn.execute(
            "SELECT 1 FROM attributes_documentation WHERE name_table=? AND attribute=?", (name_table, attribute)
        ).fetchone()
        if documented is None:
            project_conn.execute(
                "INSERT INTO attributes_documentation (name_table, attribute, description) VALUES(?,?,?)",
                (name_table, attribute, description),
            )
