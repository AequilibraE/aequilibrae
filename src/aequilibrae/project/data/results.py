import json
import logging
import uuid
from typing import Any

import pandas as pd

from aequilibrae.project.project_table import _CREATE_INDEX_SQL, NonSpatialProjectTable
from aequilibrae.utils.db_utils import NestedTransactionManager, df_sqlite_types, escape_identifier

logger = logging.getLogger(__name__)


class Results(NonSpatialProjectTable):
    """Result metadata table."""

    name = "results"
    key = "table_name"
    record_name = "ResultRecord"
    defaults = {"procedure": "", "procedure_id": "", "procedure_report": "null"}

    def __init__(
        self,
        project_connection: NestedTransactionManager,
        results_connection: NestedTransactionManager,
    ) -> None:
        """Create the result table.

        :Arguments:
            **project_connection** (:obj:`NestedTransactionManager`): Manager for
            result metadata in the project database.

            **results_connection** (:obj:`NestedTransactionManager`): Manager for
            result tables.
        """
        super().__init__(project_connection)
        self._results_connection = results_connection

    def create(
        self,
        table_name: str,
        data: pd.DataFrame,
        *,
        procedure: str | None = None,
        procedure_id: str | None = None,
        procedure_report: dict[str, Any] | None = None,
        timestamp: str | None = None,
        description: str | None = None,
        scenario: str | None = None,
        year: str | None = None,
        reference_table: str = "links",
        dtype: dict[str, str] | None = None,
    ) -> Any:
        """Create one table and its metadata record.

        :Arguments:
            **table_name** (:obj:`str`): Unique SQLite table name.

            **data** (:obj:`pandas.DataFrame`): Result values to persist.

            **procedure** (:obj:`str`, *Optional*): Producing procedure name.

            **procedure_id** (:obj:`str`, *Optional*): Producing procedure ID.

            **procedure_report** (:obj:`dict`, *Optional*): JSON-serialisable
            procedure report.

            **timestamp** (:obj:`str`, *Optional*): Result timestamp.

            **description** (:obj:`str`, *Optional*): Human-readable description.

            **scenario** (:obj:`str`, *Optional*): Scenario label.

            **year** (:obj:`str`, *Optional*): Model-year label.

            **reference_table** (:obj:`str`): Referenced project table.

            **dtype** (:obj:`dict`, *Optional*): SQLite type overrides by column.

        :Returns:
            **result record** (:obj:`Any`): Generated frozen metadata record.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if table_name in self or self._table_exists(table_name):
            raise ValueError(f"A result record or table named {table_name!r} already exists")

        frame, index_names = format_dataframe(data)
        table = escape_identifier(table_name)

        columns_to_types = df_sqlite_types(frame, dtype or {})
        columns_to_escaped = {col: escape_identifier(col) for col in columns_to_types.keys()}

        # Reorder to match
        frame = frame[list(columns_to_types.keys())]

        definitions = ", ".join(f"{columns_to_escaped[column]} {dtype}" for column, dtype in columns_to_types.items())
        placeholders = ",".join("?" for _ in columns_to_types)
        insert_sql = f"INSERT INTO {table} ({','.join(columns_to_escaped.values())}) VALUES ({placeholders})"
        index_columns = ", ".join(escape_identifier(col) for col in index_names)
        index_name = escape_identifier(f"aeq_{table_name}_idx")

        with self._results_connection.transaction() as conn:
            conn.execute(f"CREATE TABLE {table} ({definitions})")
            conn.executemany(insert_sql, frame.itertuples(index=False, name=None))
            conn.execute(_CREATE_INDEX_SQL.format(index=index_name, table=table, columns=index_columns))

        try:
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
        except BaseException as primary:
            try:
                with self._results_connection.transaction() as conn:
                    conn.execute(f"DROP TABLE {table}")
            except BaseException as cleanup:
                primary.add_note(f"unregistered result table {table_name!r} remains: {cleanup!r}")
            raise

        return self.get(table_name)

    def get_results(self, table_name: str) -> pd.DataFrame:
        """Read a result table into a DataFrame.

        :Arguments:
            **table_name** (:obj:`str`): Registered table name.

        :Returns:
            **results** (:obj:`pandas.DataFrame`): Stored result values.
        """
        record = self.get(table_name)
        return pd.read_sql_query(
            f"SELECT * FROM {escape_identifier(record.table_name)}", self._results_connection._connection
        )

    def delete_result(self, table_name: str) -> None:
        """Delete result metadata and its table.

        :Arguments:
            **table_name** (:obj:`str`): Registered result name to delete.
        """
        table = escape_identifier(table_name)
        tombstone = f"__aeq_deleted_{uuid.uuid4().hex}"
        move = self._table_exists(table_name)

        if move:
            with self._results_connection.transaction() as conn:
                conn.execute(f"ALTER TABLE {table} RENAME TO {tombstone}")

        try:
            super().delete(table_name)
        except BaseException as primary:
            if move:
                try:
                    with self._results_connection.transaction() as conn:
                        conn.execute(f"ALTER TABLE {tombstone} RENAME TO {table}")
                except BaseException as cleanup:
                    primary.add_note(f"result table is stranded as {tombstone!r}: {cleanup!r}")
            raise

        if move:
            with self._results_connection.transaction() as conn:
                conn.execute(f"DROP TABLE {tombstone}")

    def clear_database(self) -> None:
        """Remove metadata for absent tables."""
        with self._connection.transaction() as conn:
            names = [row[0] for row in conn.execute("SELECT table_name FROM results").fetchall()]
            missing = [(name,) for name in names if not self._table_exists(name)]

            if missing:
                conn.executemany("DELETE FROM results WHERE table_name=?", missing)

        self._invalidate()

    def update_database(self) -> None:
        """Register existing, unrecorded tables."""
        result_tables = {
            row[0]
            for row in self._results_connection._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        records = {row[0] for row in self._connection._connection.execute("SELECT table_name FROM results").fetchall()}
        for table_name in sorted(result_tables - records):
            self.insert(table_name=table_name)

    def sync(self) -> None:
        """Remove metadata for absent tables, and register unrecorded tables."""
        self.clear_database()
        self.update_database()

    def list(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM results", self._connection._connection)

    def _table_exists(self, table_name: str) -> bool:
        return (
            self._results_connection._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
            ).fetchone()
            is not None
        )


def format_dataframe(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if any(not isinstance(column, str) for column in data.columns):
        raise ValueError("result columns must all be strings")
    if not data.columns.is_unique:
        raise ValueError("result columns must be unique")
    names = []
    used = set(data.columns)
    for level, name in enumerate(data.index.names):
        label = name if name is not None else f"index_level_{level}"
        if not isinstance(label, str):
            raise ValueError("result index names must be strings")
        if label in used or label in names:
            raise ValueError(f"result index label {label!r} collides with another data column")
        names.append(label)
    frame = data.copy(deep=False)
    frame.index = frame.index.set_names(names)
    return frame.reset_index(), names
