import pandas as pd
import json
import sqlite3
from typing import Optional

from aequilibrae.project.network.safe_class import SafeClass


class ResultRecord(SafeClass):
    """Class for handling records of results in the AequilibraE project database.

    This class provides methods to save, delete, and retrieve result records and their data.

    Arguments:
        **data_set** (:obj:`dict`): Dictionary containing the result record data.
        **project**: (:obj:`Project`): Project object this result record belongs to.
        **project_conn** (:obj:`Optional[sqlite3.Connection])`: Connection to the project database. If None, the project's connection will be used.
        **results_conn** (:obj:`Optional[sqlite3.Connection]`): Connection to the results database. If None, the project's results connection will be used.
    """

    def __init__(
        self,
        data_set: dict,
        project,
        project_conn: Optional[sqlite3.Connection] = None,
        results_conn: Optional[sqlite3.Connection] = None,
    ):
        super().__init__(data_set, project)
        self._exists: bool
        self.__dict__["_exists"] = True

        self.__dict__["_project_conn"] = project_conn
        self.__dict__["_results_conn"] = results_conn

    def save(self) -> None:
        """Saves results record to the project database.

        Creates a new record if it doesn't exist or updates an existing one.
        """
        with self._project_conn or self.project.db_connection as conn:
            sql = "SELECT COUNT(*) FROM results WHERE table_name=?"

            if conn.execute(sql, [self.table_name]).fetchone()[0] == 0:
                data = [
                    str(self.table_name),
                    str(self.procedure),
                    str(self.procedure_id),
                    self.procedure_report,
                    str(self.timestamp),
                    str(self.description),
                    self.year,
                    self.scenario,
                    self.reference_table,
                ]
                conn.execute(
                    "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description, year, scenario, reference_table)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    data,
                )

            for key, value in self.__dict__.items():
                if key != "table_name" and key in self.__original__:
                    v_old = self.__original__.get(key, None)
                    if value != v_old and value:
                        self.__original__[key] = value
                        conn.execute(f"UPDATE results SET '{key}'=? WHERE table_name=?", [value, self.table_name])

    def delete(self) -> None:
        """Deletes this results record and the underlying data from disk.

        Removes both the record from the project database and the data table from the results database.
        """
        with (
            self._project_conn or self.project.db_connection as project_conn,
            self._results_conn or self.project.results_connection as results_conn,
        ):
            project_conn.execute("DELETE FROM results WHERE table_name=?", [self.table_name])
            results_conn.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')

        self.__dict__["_exists"] = False

    def get_data(self) -> pd.DataFrame:
        """Returns the results data for further computation.

        Returns:
            **df** (:obj:`pd.DataFrame`): DataFrame containing the results data.
        """
        with self._results_conn or self.project.results_connection as conn:
            return pd.read_sql(f'SELECT * FROM "{self.table_name}"', conn)

    def set_data(self, df: pd.DataFrame, **kwargs) -> None:
        """Set the results data corresponding to this record. Additionally saves this record.

        Additional keyword arguments forwarded to the ``pd.DataFrame.to_sql`` method.

        Arguments:
            **df** (:obj:`pd.DataFrame`): DataFrame object to save. Uses ``pd.DataFrame.to_sql``.
        """
        self.save()
        with self._results_conn or self.project.results_connection as conn:
            df.to_sql(self.table_name, conn, **kwargs)

    def __setattr__(self, instance, value) -> None:
        """Override attribute setting with validation.

        Validates table_name uniqueness and handles report serialisation.

        Arguments:
            **instance** (:obj:`str`): Attribute name.
            **value** (:obj:`object`): Value to set.

        Raises:
            ValueError: If trying to set a table_name that already exists.
        """
        if instance == "table_name":
            with self._project_conn or self.project.db_connection as conn:
                sql = f"Select count(*) from results where {instance}=?"
                qry_value = sum(conn.execute(sql, [str(value)]).fetchone())
                if qry_value > 0:
                    raise ValueError("Another results with this table_name already exists")

        self.__dict__[instance] = value
