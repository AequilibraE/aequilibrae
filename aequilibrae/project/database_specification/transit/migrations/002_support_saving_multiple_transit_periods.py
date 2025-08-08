import sqlite3
import pathlib
from typing import Optional

from aequilibrae import Project, logger
from aequilibrae.context import get_active_project
from aequilibrae.project.project_creation import add_triggers, remove_triggers, recreate_columns


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    logger.info("Beginning migration to support saving and loading multiple Transit graphs")
    project: Project = get_active_project(must_exist=True)

    if not transit_conn:
        logger.info("Migration finished, no 'public_transport.sqlite' connection provided.")
        return

    links_schema = pathlib.Path(__file__).parent.parent / "tables" / "links.sql"
    if not links_schema.exists():
        raise FileNotFoundError(str(links_schema))

    nodes_schema = pathlib.Path(__file__).parent.parent / "tables" / "nodes.sql"
    if not nodes_schema.exists():
        raise FileNotFoundError(str(nodes_schema))

    try:
        with project.db_connection as project_conn:
            period_ids = project_conn.execute("SELECT period_id FROM transit_graph_configs").fetchall()
    except sqlite3.OperationalError:
        logger.info("Migration finished, no 'transit_graph_configs' table found.")
        return

    existing_links = transit_conn.execute("SELECT link_id FROM links LIMIT 1").fetchone()

    if len(period_ids) > 1:
        raise ValueError(
            "more than one period_id found in 'transit_graph_configs' cannot migrate with multiple possible period_ids"
        )
    elif len(period_ids) == 0:
        if existing_links is not None:
            raise ValueError("no period_id found in 'transit_graph_configs' cannot migrate with without period_id")
        else:
            period_id = project.network.periods.default_period.period_id
    else:
        period_id = period_ids[0][0]

    with open(links_schema, "r") as links_sql, open(nodes_schema, "r") as nodes_sql:
        sqls = [
            "DROP INDEX idx_link",
            "DROP INDEX idx_link_anode",
            "DROP INDEX idx_link_bnode",
            "DROP INDEX idx_link_modes",
            "DROP INDEX idx_link_link_type",
            "DROP INDEX idx_links_a_node_b_node",
            "SELECT RenameTable(NULL, 'links', '__old_links')",
            "DROP INDEX idx_node",
            "DROP INDEX idx_node_is_centroid",
            "SELECT RenameTable(NULL, 'nodes', '__old_nodes')",
            *links_sql.read().split("--#"),
            *nodes_sql.read().split("--#"),
        ]

    remove_triggers(transit_conn, logger, "transit")
    try:
        for sql in sqls:
            transit_conn.execute(sql)

        for table in ["links", "nodes"]:
            columns = recreate_columns(transit_conn, logger, table, f"__old_{table}")

            transit_conn.execute(
                f"""INSERT INTO {table}({",".join(columns)},'period_id') SELECT {",".join(columns)},{period_id} FROM __old_{table}"""
            )
            transit_conn.execute(f"SELECT DropTable(NULL, '__old_{table}')")

    finally:
        add_triggers(transit_conn, logger, "transit")

    logger.info("Migration successful")
