import sqlite3
import pathlib
from typing import Optional

from aequilibrae import Project, logger
from aequilibrae.transit import Transit
from aequilibrae.context import get_active_project
from aequilibrae.project.project_creation import add_triggers, remove_triggers, recreate_columns

import numpy as np


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    logger.info("Beginning migration to align taz_ids and node_ids for origins/destinations/centroids")

    if not transit_conn:
        logger.info("Migration finished, no 'public_transport.sqlite' connection provided.")
        return

    project: Project = get_active_project(must_exist=True)

    nodes_schema = pathlib.Path(__file__).parent.parent / "tables" / "nodes.sql"
    if not nodes_schema.exists():
        raise FileNotFoundError(str(nodes_schema))

    logger.info("Loading PT graphs...")
    data: Transit = Transit(project)
    try:
        data.load()
    except sqlite3.OperationalError:
        logger.info("Migration finished, no 'transit_graph_configs' table found.")
        return

    with open(nodes_schema, "r") as nodes_sql:
        sqls = [
            "DELETE FROM links",
            "DROP INDEX idx_node",
            "DROP INDEX idx_period_nodes",
            "DROP INDEX idx_node_is_centroid",
            "SELECT RenameTable(NULL, 'nodes', '__old_nodes')",
            *nodes_sql.read().split("--#"),
        ]

    logger.info("Removing triggers...")
    remove_triggers(transit_conn, logger, "transit")
    try:
        logger.info("Removing/renaming tables...")
        for sql in sqls:
            transit_conn.execute(sql)

        for graph_builder in data.graphs.values():
            logger.info(f"Aligning graph for period {graph_builder.period_id}...")
            graph_builder.vertices.loc[graph_builder.vertices.taz_id == "", "taz_id"] = -1
            graph_builder.vertices.taz_id = graph_builder.vertices.taz_id.astype("int64")

            o_vertices = graph_builder.vertices[
                (graph_builder.vertices.taz_id > 0) & (graph_builder.vertices.node_type.isin(["origin", "od"]))
            ]

            node_id_min = o_vertices.taz_id.max() + 1
            graph_builder.vertices["__new_node_id"] = np.hstack(
                (
                    o_vertices.taz_id.to_numpy(),
                    np.arange(node_id_min, node_id_min + len(graph_builder.vertices) - len(o_vertices)),
                )
            )

            vertices = graph_builder.vertices[["node_id", "__new_node_id"]]
            graph_builder.edges = (
                graph_builder.edges.merge(vertices, left_on="a_node", right_on="node_id", validate="m:1")
                .drop(columns=["node_id", "a_node"])
                .rename(columns={"__new_node_id": "a_node"})
                .merge(vertices, left_on="b_node", right_on="node_id", validate="m:1")
                .drop(columns=["node_id", "b_node"])
                .rename(columns={"__new_node_id": "b_node"})
            )

            graph_builder.vertices = graph_builder.vertices.drop(columns="node_id").rename(
                columns={"__new_node_id": "node_id"}
            )

            logger.info(f"Saving graph for period {graph_builder.period_id}...")
            graph_builder.save(pt_conn=transit_conn)

        transit_conn.execute("SELECT DropTable(NULL, '__old_nodes')")

    finally:
        logger.info("Re-adding triggers...")
        add_triggers(transit_conn, logger, "transit")

    logger.info("Migration successful")
