import sqlite3
import pathlib

from aequilibrae import Project, logger
from aequilibrae.context import get_active_project
from aequilibrae.project.project_creation import add_triggers, remove_triggers
from aequilibrae.project.database_connection import database_connection


def migrate(conn: sqlite3.Connection):
    logger.info("Beginning migration to support saving and loading multiple Transit graphs")
    project: Project = get_active_project(must_exist=True)

    if not (project.project_base_path / "public_transport.sqlite").exists():
        logger.info("Migration finished, no 'public_transport.sqlite' found.")
        return

    links_schema = pathlib.Path(__file__).parent.parent / "tables" / "links.sql"
    if not links_schema.exists():
        raise FileNotFoundError(str(links_schema))

    nodes_schema = pathlib.Path(__file__).parent.parent / "tables" / "nodes.sql"
    if not nodes_schema.exists():
        raise FileNotFoundError(str(nodes_schema))

    with project.db_connection as project_conn:
        period_ids = project_conn.execute("SELECT period_id FROM transit_graph_configs").fetchall()

    existing_links = conn.execute("SELECT link_id FROM links LIMIT 1").fetchone()

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

    remove_triggers(conn, logger, "transit")
    try:
        for sql in sqls:
            conn.execute(sql)

        for table in ["links", "nodes"]:
            columns = conn.execute(
                f"SELECT name, type FROM PRAGMA_TABLE_INFO('__old_{table}') AS table_info"
            ).fetchall()
            columns = {f"{x[0]}": x[1] for x in columns if x[0]}

            orig_columns = conn.execute(f"SELECT name, type FROM PRAGMA_TABLE_INFO('{table}') AS table_info").fetchall()
            orig_columns = {f"{x[0]}" for x in orig_columns}

            new_columns = {k: v for k, v in columns.items() if k not in orig_columns}
            sql = "ALTER TABLE {} ADD COLUMN {} {};"
            for k, v in new_columns.items():
                conn.execute(sql.format(table, k, v))

            conn.execute(
                f"""INSERT INTO {table}({",".join(columns)},'period_id') SELECT {",".join(columns)},{period_id} FROM __old_{table}"""
            )
            conn.execute(f"SELECT DropTable(NULL, '__old_{table}')")

    finally:
        add_triggers(conn, logger, "transit")

    logger.info("Migration successful")
