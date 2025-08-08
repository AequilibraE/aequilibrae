import sqlite3
import json
from typing import Optional

from aequilibrae import Project, logger
from aequilibrae.context import get_active_project
from aequilibrae.project.data import Results

import pandas as pd


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    logger.info("Beginning migration to move transit results to the main project_database.sqlite")

    if not transit_conn:
        logger.info("Migration finished, no 'public_transport.sqlite' connection provided.")
        return
    elif not results_conn:
        logger.info("Migration finished, no 'results.sqlite' connection provided.")
        return

    if transit_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'").fetchone() is None:
        logger.info("Migration finished, no table 'results' in 'public_transport.sqlite'.")
        return

    project: Project = get_active_project(must_exist=True)

    results = Results(project, project_conn=project_conn, results_conn=results_conn)
    results.update_database()

    transit_results = Results(project, project_conn=transit_conn, results_conn=results_conn)
    df = transit_results.list()

    for _, row in df.iterrows():
        logger.info(f"Migrating the {row.table_name} results record")
        record = results.get_record(row.table_name)

        record.procedure = row.procedure
        record.procedure_id = row.procedure_id
        record.procedure_report = row.procedure_report
        record.timestamp = row.timestamp
        record.description = row.description

        record.save()

    logger.info("Dropping the transit results table")
    transit_conn.execute("DROP TABLE IF EXISTS results")
