import os
from os.path import isfile, join
import json

import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data.result_record import ResultRecord
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.utils.db_utils import add_column_unless_exists, commit_and_close


class Results:
    """Gateway into the results available/recorded in the model"""

    def __init__(self, project):
        self.project = project
        self.logger = project.logger
        self.__items = {}
        self.__fields = []

        tl = TableLoader()
        with self.project.db_connection as conn:
            results_list = tl.load_table(conn, "results")
        self.__fields = list(tl.fields)
        if results_list:
            self.__properties = list(results_list[0].keys())

        with self.project.db_connection as conn:
            for lt in results_list:
                table_name = lt["table_name"]
                if table_name in self.__items:
                    if not self.__items[table_name]._exists:
                        del self.__items[table_name]

                if table_name not in self.__items:
                    if conn.execute("SELECT COUNT(*) FROM results WHERE table_name=?", (table_name,)).fetchone()[0]:
                        self.__items[table_name] = ResultRecord(lt, project)

    def reload(self):
        """Reloads the results from the database"""
        self.__items.clear()
        self.__init__(self.project)

    def clear_database(self) -> None:
        """Removes records from the results table that do not exist in the results database"""

        with self.project.db_connection as project_conn, self.project.results_connection as results_conn:
            mats = [x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()]

            remove = set(mats) - {
                name
                for name in mats
                if results_conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
                ).fetchone()
                is not None
            }
            if remove:
                self.logger.warning(f"Results records not found in results database: {','.join(remove)}")

                project_conn.executemany("DELETE FROM results WHERE table_name=?;", [(x,) for x in remove])

    def update_database(self) -> None:
        """Adds records to the results table for results found in the results database"""
        with self.project.db_connection as project_conn, self.project.results_connection as results_conn:
            existing_results = {x[0] for x in results_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            existing_records = {x[0] for x in project_conn.execute("SELECT table_name FROM results").fetchall()}

        new_results = existing_results - existing_records

        if new_results:
            self.logger.warning(f"New results found in the results database. Added to the database: {','.join(new_results)}")

        for table in new_results:
            rec = self.new_record(table)
            rec.save()

    def list(self) -> pd.DataFrame:
        """List of all results available

        :Returns:
            **df** (:obj:`pd.DataFrame`): Pandas DataFrame listing all results available in the model
        """

        with self.project.db_connection as conn:
            return pd.read_sql_query("SELECT * FROM results;", conn)

    def get_results(self, table_name: str) -> pd.DataFrame:
        """Returns an DataFrame containing the results

        Raises an error if results does not exist

        :Arguments:
            **table_name** (:obj:`str`): Name of the results to be loaded

        :Returns:
            **results** (:obj:`pd.DataFrame`): Results objects
        """

        return self.get_record(table_name).get_data()

    def get_record(self, table_name: str) -> ResultRecord:
        """Returns a model ResultsRecord for manipulation in memory"""

        if table_name.lower() not in self.__items:
            raise Exception("There is no results record with that name")

        if not self.__items[table_name.lower()]._exists:
            raise Exception("This result was deleted during this session")

        return self.__items[table_name.lower()]

    def check_exists(self, table_name: str) -> bool:
        """Checks whether a result with a given name exists

        :Returns:
            **exists** (:obj:`bool`): Does the matrix exist?
        """
        return table_name.lower() in self.__items

    def delete_record(self, table_name: str) -> None:
        """Deletes a ResultRecord from the model and attempts to remove from it from the results database"""
        rr = self.get_record(table_name)
        rr.delete()

    def new_record(
            self,
            table_name: str,
            procedure: str = None,
            procedure_id: str = None,
            procedure_report: dict = None,
            timestamp: str = None,
            description: str = None,
        ) -> ResultRecord:
        """Creates a new record for a result.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table
            **procedure** (:obj:`str`, optional): Name of the procedure
            **procedure_id** (:obj:`str`, optional): ID of the procedure
            **procedure_report** (:obj:`str`, optional): Report associated with the procedure
            **timestamp** (:obj:`str`, optional): Timestamp for the record
            **description** (:obj:`str`, optional): Description of the record

        :Returns:
            **result_record** (:obj:`ResultRecord`): A result record that can be manipulated in memory before saving
        """
        if table_name in self.__items:
            raise ValueError(f"There is already a result of name ({table_name}). It must be unique.")

        tp = {
            "table_name": table_name,
            "procedure": procedure,
            "procedure_id": procedure_id,
            "procedure_report": json.dumps(procedure_report),
            "timestamp": timestamp,
            "description": description,
        }
        rr = ResultRecord(tp, self.project)
        rr.save()
        self.__items[table_name.lower()] = rr
        self.logger.warning("ResultRecord has been saved to the database")
        return rr

    def _clear(self):
        """Eliminates records from memory. For internal use only"""
        self.__items.clear()
