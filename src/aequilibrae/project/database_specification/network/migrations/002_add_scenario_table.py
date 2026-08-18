import pathlib

from aequilibrae.log import logger
from aequilibrae.project.project_creation import run_queries_from_sql_file


def migrate(*, closure):
    logger.info("Beginning migration to add scenario support to the project database")
    schema = pathlib.Path(__file__).parent.parent / "tables" / "scenarios.sql"
    run_queries_from_sql_file(closure["project"], schema)
