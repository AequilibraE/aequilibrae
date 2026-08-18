import datetime
import json
import logging
import uuid

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from aequilibrae.project.project_table import ProjectTable

logger = logging.getLogger(__name__)


class Results(ProjectTable):
    """Result metadata gateway with compensated payload-table helpers."""

    name = "results"
    key = "table_name"
    record_name = "ResultRecord"
    defaults = {"procedure": "", "procedure_id": "", "procedure_report": "null"}

    def __init__(self, project_transactions, results_transactions):
        super().__init__(project_transactions)
        self._results_transactions = results_transactions

    def create(
        self,
        table_name: str,
        data: pd.DataFrame,
        *,
        procedure: str = None,
        procedure_id: str = None,
        procedure_report: dict = None,
        timestamp: str = None,
        description: str = None,
        scenario: str = None,
        year: str = None,
        reference_table: str = "links",
        dtype: dict = None,
        chunksize: int = 1000,
    ):
        """Create one fail-if-present payload table and its metadata record."""
        self._require_resource_idle()
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer")
        if table_name in self or self._payload_exists(table_name):
            raise ValueError(f"A result metadata record or payload table named {table_name!r} already exists")

        frame = _payload_frame(data)
        types = _sqlite_types(frame, dtype or {})
        table = _quote_identifier(table_name)
        columns = list(frame.columns)
        definitions = ", ".join(
            f"{_quote_identifier(column)} {types[column]}" for column in columns
        )
        placeholders = ",".join("?" for _ in columns)
        insert_sql = (
            f"INSERT INTO {table} ({','.join(_quote_identifier(column) for column in columns)}) "
            f"VALUES ({placeholders})"
        )

        with self._results_transactions.transaction():
            self._results_transactions.execute(f"CREATE TABLE {table} ({definitions})")
            for chunk in _row_chunks(frame, chunksize):
                self._results_transactions.executemany(insert_sql, chunk)

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
                with self._results_transactions.transaction():
                    self._results_transactions.execute(f"DROP TABLE {table}")
            except BaseException as cleanup:
                _add_resource_note(primary, f"unregistered payload table {table_name!r} remains: {cleanup!r}")
            raise
        return self.get(table_name)

    def get_results(self, table_name: str) -> pd.DataFrame:
        record = self.get(table_name)
        return pd.read_sql_query(f"SELECT * FROM {_quote_identifier(record.table_name)}", self._results_transactions)

    def delete_result(self, table_name: str):
        """Delete both result metadata and its payload table."""
        self._require_resource_idle()
        table = _quote_identifier(table_name)
        tombstone_name = f"__aeq_deleted_{uuid.uuid4().hex}"
        tombstone = _quote_identifier(tombstone_name)
        moved = self._payload_exists(table_name)
        if moved:
            with self._results_transactions.transaction():
                self._results_transactions.execute(f"ALTER TABLE {table} RENAME TO {tombstone}")
        try:
            super().delete(table_name)
        except BaseException as primary:
            if moved:
                try:
                    with self._results_transactions.transaction():
                        self._results_transactions.execute(f"ALTER TABLE {tombstone} RENAME TO {table}")
                except BaseException as cleanup:
                    _add_resource_note(primary, f"payload is stranded as {tombstone_name!r}: {cleanup!r}")
            raise
        if moved:
            with self._results_transactions.transaction():
                self._results_transactions.execute(f"DROP TABLE {tombstone}")

    def clear_database(self) -> None:
        """Remove metadata for absent payloads without changing payload tables."""
        with self._transactions.transaction():
            names = [row[0] for row in self._transactions.execute("SELECT table_name FROM results").fetchall()]
            missing = [(name,) for name in names if not self._payload_exists(name)]
            if missing:
                self._transactions.executemany("DELETE FROM results WHERE table_name=?", missing)
        self._invalidate()

    def update_database(self) -> None:
        """Register existing, unowned payload tables as metadata only."""
        payloads = {
            row[0]
            for row in self._results_transactions.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        records = {row[0] for row in self._transactions.execute("SELECT table_name FROM results").fetchall()}
        for table_name in sorted(payloads - records):
            self.insert(table_name=table_name)

    def list(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM results", self._transactions)

    def _payload_exists(self, table_name):
        return self._results_transactions.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone() is not None

    def _require_resource_idle(self):
        if self._transactions.in_transaction or self._results_transactions.in_transaction:
            raise RuntimeError("result payload helpers cannot run inside a database transaction")


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("SQLite identifiers must be non-empty strings")
    return '"' + identifier.replace('"', '""') + '"'


def _payload_frame(data: pd.DataFrame) -> pd.DataFrame:
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
            raise ValueError(f"result index label {label!r} collides with another payload column")
        names.append(label)
    frame = data.copy(deep=False)
    frame.index = frame.index.set_names(names)
    return frame.reset_index()


def _sqlite_types(frame: pd.DataFrame, overrides: dict) -> dict:
    unknown = set(overrides) - set(frame.columns)
    if unknown:
        raise ValueError(f"dtype overrides refer to unknown columns: {sorted(unknown)}")
    result = {}
    for column in frame.columns:
        if column in overrides:
            value = overrides[column]
            normalized = value.upper() if isinstance(value, str) else None
            if normalized not in {"INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC"}:
                raise ValueError(f"invalid SQLite dtype for {column!r}")
            result[column] = normalized
        elif ptypes.is_bool_dtype(frame[column].dtype) or ptypes.is_integer_dtype(frame[column].dtype):
            result[column] = "INTEGER"
        elif ptypes.is_float_dtype(frame[column].dtype):
            result[column] = "REAL"
        elif ptypes.is_object_dtype(frame[column].dtype) and all(
            isinstance(value, bytes) for value in frame[column].dropna()
        ):
            result[column] = "BLOB"
        else:
            result[column] = "TEXT"
    return result


def _row_chunks(frame: pd.DataFrame, chunksize: int):
    chunk = []
    for row in frame.itertuples(index=False, name=None):
        chunk.append(tuple(_sqlite_value(value) for value in row))
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _sqlite_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        return value.total_seconds()
    if isinstance(value, (str, bytes, int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    raise TypeError(f"value {value!r} cannot be stored in SQLite")


def _add_resource_note(error: BaseException, message: str):
    if hasattr(error, "add_note"):
        error.add_note(message)
    else:  # pragma: no cover
        logger.error(message)
