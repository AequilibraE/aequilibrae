import json
import logging
import sqlite3
from contextlib import nullcontext
from typing import Optional

import pandas as pd

from aequilibrae.project.project_table import ProjectTable

logger = logging.getLogger(__name__)


class Results(ProjectTable):
    """Gateway into the results available/recorded in the model

    Result metadata lives in the *results* table of the project database, and
    the data itself in the results database (``results_database.sqlite``).

    .. code-block:: python

        >>> results = project.results  # doctest: +SKIP

        # Record and store a new result in one go
        >>> results.create("assignment_2026", df, procedure="traffic assignment")  # doctest: +SKIP

        # and read it back
        >>> df = results.get_results("assignment_2026")  # doctest: +SKIP
    """

    name = "results"
    key = "table_name"
    record_name = "ResultRecord"
    #: the schema requires these; "" / JSON null mean "not provided"
    defaults = {"procedure": "", "procedure_id": "", "procedure_report": "null"}

    def __init__(
        self,
        project,
        project_conn: Optional[sqlite3.Connection] = None,
        results_conn: Optional[sqlite3.Connection] = None,
    ):
        """Initialise the Results object.

        :Arguments:
            **project**: Project instance this Results object belongs to

            **project_conn** (:obj:`Optional[sqlite3.Connection]`): Optional connection to the
            database holding the results table.

            **results_conn** (:obj:`Optional[sqlite3.Connection]`): Optional connection to the
            results database
        """
        super().__init__(project)
        self.__project_conn = project_conn
        self.__results_conn = results_conn

    def create(
        self,
        table_name: str,
        data: pd.DataFrame = None,
        *,
        procedure: str = None,
        procedure_id: str = None,
        procedure_report: dict = None,
        timestamp: str = None,
        description: str = None,
        scenario: str = None,
        year: str = None,
        reference_table: str = "links",
        **to_sql,
    ):
        """Creates a result record and, if data is given, stores it in the results database

        :Arguments:
            **table_name** (:obj:`str`): Name for the result. Must be unique

            **data** (:obj:`pd.DataFrame`, *Optional*): Result data, written to the results
            database via ``pd.DataFrame.to_sql``. Extra keyword arguments are forwarded to it

            **procedure**, **procedure_id**, **procedure_report**, **timestamp**, **description**,
            **scenario**, **year**, **reference_table**: Metadata for the record

        :Returns:
            **result record**: The record for the new result
        """
        if table_name in self:
            raise ValueError(f"There is already a result of name ({table_name}). It must be unique.")

        self.insert(
            table_name=table_name,
            procedure=procedure,
            procedure_id=procedure_id,
            procedure_report=json.dumps(procedure_report),
            timestamp=timestamp,
            description=description,
            scenario=scenario,
            year=year,
            reference_table=reference_table,
        )

        if data is not None:
            with self._results_ctx() as conn:
                data.to_sql(table_name, conn, **to_sql)

        return self.get(table_name)

    def get_results(self, table_name: str) -> pd.DataFrame:
        """Returns the data stored for one result

        Raises an error if the result record does not exist.

        :Arguments:
            **table_name** (:obj:`str`): Name of the result to be loaded

        :Returns:
            **results** (:obj:`pd.DataFrame`): Results as a DataFrame
        """
        record = self.get(table_name)
        with self._results_ctx() as conn:
            return pd.read_sql(f'SELECT * FROM "{record.table_name}"', conn)

    def delete(self, table_name: str, conn=None):
        """Deletes a result record and drops its table from the results database"""
        super().delete(table_name, conn=conn)
        with self._results_ctx() as results_conn:
            results_conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    def clear_database(self) -> None:
        """Removes records from the results table that do not exist in the results database."""

        with self._write_ctx(None) as project_conn, self._results_ctx() as results_conn:
            recorded = [x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()]

            remove = {
                name
                for name in recorded
                if results_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
                is None
            }
            if remove:
                logger.warning(f"Results records not found in results database: {','.join(remove)}")

                project_conn.executemany("DELETE FROM results WHERE table_name=?;", [(x,) for x in remove])
            else:
                logger.info("No result records to remove")

    def update_database(self) -> None:
        """Adds records to the results table for results found in the results database."""
        with self._read_ctx(None) as project_conn, self._results_ctx() as results_conn:
            existing_results = {
                x[0] for x in results_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            existing_records = {x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()}

        new_results = existing_results - existing_records

        if new_results:
            logger.warning(f"New results found in the results database. Added to the database: {','.join(new_results)}")
            for table in new_results:
                self.create(table)
        else:
            logger.info("No new result records to add")

    def list(self) -> pd.DataFrame:
        """List of all results available.

        :Returns:
            **df** (:obj:`pd.DataFrame`): Pandas DataFrame listing all results available in the model
        """
        with self._read_ctx(None) as conn:
            return pd.read_sql_query("SELECT * FROM results;", conn)

    def _read_ctx(self, conn):
        return super()._read_ctx(conn if conn is not None else self.__project_conn)

    def _write_ctx(self, conn):
        return super()._write_ctx(conn if conn is not None else self.__project_conn)

    def _results_ctx(self):
        if self.__results_conn is not None:
            return nullcontext(self.__results_conn)
        return self.project.results_connection
