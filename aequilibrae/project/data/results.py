import json
import sqlite3
from typing import Optional

import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data.result_record import ResultRecord
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.utils.db_utils import add_column_unless_exists, commit_and_close


class Results:
    """Gateway into the results available/recorded in the model"""

    def __init__(
        self,
        project,
        project_conn: Optional[sqlite3.Connection] = None,
        results_conn: Optional[sqlite3.Connection] = None,
    ):
        """Initialise the Results object.

        Arguments:
            **project**: Project instance this Results object belongs to
            **project_conn** (:obj:`Optional[sqlite3.Connection]`): Optional connection to the database to use for the results table.
            **results_conn** (:obj:`Optional[sqlite3.Connection]`): Optional connection to the results database
        """
        self.project = project
        self.logger = project.logger
        self.__items = {}
        self.__fields = []

        self.__project_conn = project_conn
        self.__results_conn = results_conn

        tl = TableLoader()
        with self.__project_conn or self.project.db_connection as conn:
            results_list = tl.load_table(conn, "results")
        self.__fields = list(tl.fields)
        if results_list:
            self.__properties = list(results_list[0].keys())

        with self.__project_conn or self.project.db_connection as conn:
            for lt in results_list:
                table_name = lt["table_name"]
                if table_name in self.__items:
                    if not self.__items[table_name]._exists:
                        del self.__items[table_name]

                if table_name not in self.__items:
                    if conn.execute("SELECT COUNT(*) FROM results WHERE table_name=?", (table_name,)).fetchone()[0]:
                        self.__items[table_name] = ResultRecord(
                            lt, project, project_conn=self.__project_conn, results_conn=self.__results_conn
                        )

    def reload(self) -> None:
        """Reloads the results from the database."""
        self.__items.clear()
        self.__init__(self.project, self.__project_conn, self.__results_conn)

    def clear_database(self) -> None:
        """Removes records from the results table that do not exist in the results database."""

        with (
            self.__project_conn or self.project.db_connection as project_conn,
            self.__results_conn or self.project.results_connection as results_conn,
        ):
            mats = [x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()]

            remove = set(mats) - {
                name
                for name in mats
                if results_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
                is not None
            }
            if remove:
                self.logger.warning(f"Results records not found in results database: {','.join(remove)}")

                project_conn.executemany("DELETE FROM results WHERE table_name=?;", [(x,) for x in remove])
            else:
                self.logger.info("No result records to remove")

    def update_database(self) -> None:
        """Adds records to the results table for results found in the results database."""
        with (
            self.__project_conn or self.project.db_connection as project_conn,
            self.__results_conn or self.project.results_connection as results_conn,
        ):
            existing_results = {
                x[0] for x in results_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            existing_records = {x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()}

        new_results = existing_results - existing_records

        if new_results:
            self.logger.warning(
                f"New results found in the results database. Added to the database: {','.join(new_results)}"
            )
            for table in new_results:
                rec = self.new_record(table)
                rec.save()
        else:
            self.logger.info("No new result records to add")

    def list(self) -> pd.DataFrame:
        """List of all results available.

        Arguments:
            **conn** (:obj:`Optional[sqlite3.Connection]`): Optional connection to use

        Returns:
            **df** (:obj:`pd.DataFrame`): Pandas DataFrame listing all results available in the model
        """

        with self.__project_conn or self.project.db_connection as conn:
            return pd.read_sql_query("SELECT * FROM results;", conn)

    def get_results(self, table_name: str) -> pd.DataFrame:
        """Returns a DataFrame containing the results.

        Raises an error if results do not exist.

        Arguments:
            **table_name** (:obj:`str`): Name of the results to be loaded

        Returns:
            **results** (:obj:`pd.DataFrame`): Results as a DataFrame

        Raises:
            **ValueError**: If the result doesn't exist
        """

        return self.get_record(table_name).get_data()

    def get_record(self, table_name: str) -> ResultRecord:
        """Returns a model ResultsRecord for manipulation in memory.

        Arguments:
            **table_name** (:obj:`str`): Name of the result record to retrieve

        Returns:
            **record** (:obj:`ResultRecord`): The requested result record

        Raises:
            **ValueError**: If the result doesn't exist or was deleted
        """

        if table_name not in self.__items:
            raise ValueError("There is no results record with that name")

        if not self.__items[table_name]._exists:
            raise ValueError("This result was deleted during this session")

        return self.__items[table_name]

    def check_exists(self, table_name: str) -> bool:
        """Checks whether a result with a given name exists.

        Arguments:
            **table_name** (:obj:`str`): Name of the result to check

        Returns:
            **exists** (:obj:`bool`): Does the result exist?
        """
        return table_name in self.__items and self.__items[table_name]._exists

    def delete_record(self, table_name: str) -> None:
        """Deletes a ResultRecord from the model and attempts to remove it from the results database.

        Arguments:
            **table_name** (:obj:`str`): Name of the result to delete

        Raises:
            **ValueError**: If the result doesn't exist
        """
        rr = self.get_record(table_name)
        rr.delete()
        del self.__items[table_name]

    def new_record(
        self,
        table_name: str,
        procedure: str = None,
        procedure_id: str = None,
        procedure_report: dict = None,
        timestamp: str = None,
        description: str = None,
        scenario: str = None,
        year: str = None,
        reference_table: str = "links",
    ) -> ResultRecord:
        """Creates a new record for a result.

        Arguments:
            **table_name** (:obj:`str`): Name of the table
            **procedure** (:obj:`str`, optional): Name of the procedure
            **procedure_id** (:obj:`str`, optional): ID of the procedure
            **procedure_report** (:obj:`dict`, optional): Report associated with the procedure
            **timestamp** (:obj:`str`, optional): Timestamp for the record
            **description** (:obj:`str`, optional): Description of the record

        Returns:
            **result_record** (:obj:`ResultRecord`): A result record that can be manipulated in memory before saving

        Raises:
            **ValueError**: If a result with the same name already exists
        """
        if table_name in self.__items:
            raise ValueError(f"There is already a result of name ({table_name}). It must be unique.")

        tp = {
            "scenario": scenario,
            "year": year,
            "table_name": table_name,
            "procedure": procedure,
            "procedure_id": procedure_id,
            "procedure_report": json.dumps(procedure_report),
            "timestamp": timestamp,
            "description": description,
            "reference_table": reference_table,
        }
        rr = ResultRecord(tp, self.project, project_conn=self.__project_conn, results_conn=self.__results_conn)
        rr.save()
        self.__items[table_name] = rr
        self.logger.warning("ResultRecord has been saved to the database")
        return rr

    def _clear(self) -> None:
        """Eliminates records from memory. For internal use only."""
        self.__items.clear()
