"""Table objects backed by a scenario's persistent SQLite connection."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass
from functools import lru_cache
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely.wkb
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.utils.db_utils import NestedTransactionManager, escape_identifier

_TABLE_INFO_SQL = 'PRAGMA table_info("{table}")'
_SCHEMA_VERSION_SQL = "PRAGMA schema_version"
_SELECT_SQL = 'SELECT {columns} FROM "{table}"'
_SELECT_ONE_SQL = 'SELECT {columns} FROM "{table}" WHERE "{key}"=? LIMIT 1'
_COUNT_SQL = 'SELECT COUNT(*) FROM "{table}"'
_CONTAINS_SQL = 'SELECT 1 FROM "{table}" WHERE "{key}"=? LIMIT 1'
_DELETE_SQL = 'DELETE FROM "{table}" WHERE "{key}"=?'
_MAX_KEY_SQL = 'SELECT MAX("{key}") FROM "{table}"'
_CHANGE_KEY_SQL = 'UPDATE "{table}" SET "{key}"=? WHERE "{key}"=?'
_EXTENT_SQL = 'SELECT ST_AsBinary(GetLayerExtent("{table}"))'
_INSERT_SQL = 'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})'
_UPDATE_SQL = 'UPDATE "{table}" SET {assignments} WHERE "{key}"=?'
_NON_EXISTANT_ID_SQL = (
    "WITH _temp_subset(id) AS (VALUES {{values}}) "  # double braces so we can format twice
    "SELECT _temp_subset.id FROM _temp_subset "
    'WHERE _temp_subset.id NOT IN (SELECT "{key}" FROM "{table}")'
)
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS {index} ON {table} ({columns})"
_VALUE_PLACEHOLDER = "?"
_GEOMETRY_COLUMN = 'ST_AsBinary("geometry") AS "geometry"'
_ASSIGNMENT = '"{column}"={placeholder}'
_GEOMETRY_PLACEHOLDER = "GeomFromWKB(?, {srid})"
_MULTI_GEOMETRY_PLACEHOLDER = "ST_Multi(GeomFromWKB(?, {srid}))"


_SQLITE_TYPE_HINTS = (
    ("INT", int),
    ("CHAR", str),
    ("CLOB", str),
    ("TEXT", str),
    ("BLOB", bytes),
    ("REAL", float),
    ("FLOA", float),  # Not typo see https://www.sqlite.org/datatype3.html
    ("DOUB", float),  # Not typo
    ("BOOL", bool),
    ("DATE", str),
    ("TIME", str),
    ("NUMERIC", float),
    ("DECIMAL", float),
)

MISSING = object()


def _python_type(column: str, declared_type: str, nullable: bool) -> type[Any]:
    """Return a best-effort Python annotation for one SQLite column."""
    if column == "geometry":
        hint: type[Any] = BaseGeometry
    else:
        hint = next((hint for token, hint in _SQLITE_TYPE_HINTS if token in declared_type.upper()), Any)
    return hint | None if nullable and hint is not Any else hint


def guess_record_type(connection: NestedTransactionManager, table: str, record_name: str) -> type[Any]:
    """Build a frozen record type from the table's current schema."""
    schema = connection._connection.execute(_TABLE_INFO_SQL.format(table=table)).fetchall()
    record_fields = []
    for _, column, declared_type, required, _, primary_key in schema:
        if column == "ogc_fid":
            continue
        record_fields.append((column, _python_type(column, declared_type, not required and not primary_key)))
    return make_dataclass(record_name, record_fields, frozen=True)


@lru_cache
def _format_insert(table: str, columns: tuple[str, ...], geometry_placeholder: str | None) -> str:
    """Format one INSERT shape, shared by every table instance."""
    names = ",".join(escape_identifier(column) for column in columns)
    placeholders = ",".join(
        geometry_placeholder if column == "geometry" and geometry_placeholder else _VALUE_PLACEHOLDER
        for column in columns
    )
    return _INSERT_SQL.format(table=table, columns=names, placeholders=placeholders)


@lru_cache
def _format_update(table: str, key: str, columns: tuple[str, ...], geometry_placeholder: str | None) -> str:
    """Format one UPDATE shape, shared by every table instance."""
    assignments = []
    for column in columns:
        placeholder = geometry_placeholder if column == "geometry" and geometry_placeholder else _VALUE_PLACEHOLDER
        assignments.append(_ASSIGNMENT.format(column=column, placeholder=placeholder))
    return _UPDATE_SQL.format(table=table, assignments=",".join(assignments), key=key)


class ProjectTable(ABC):
    """
    Common implementation for a database table.

    Subclasses declare the table name, key, and generated-record name. During construction, ``record_type`` a frozen
    dataclass matching the SQLite schema, including user-added fields, is created. Children must inherit from either
    :class:`NonSpatialProjectTable` or :class:`SpatialProjectTable`.

    Bulk operations create a new transaction, single inserts or updates use a no-op transaction if one is already open,
    other they create one.
    """

    name: str = ""
    key: str = ""
    record_name: str = ""
    record_type: type[Any]
    defaults: Mapping[str, Any] = {}
    _geometry_placeholder: str | None = None
    has_numeric_key = False

    def __init__(self, connection: NestedTransactionManager) -> None:
        """Configure the table and pre-format its SQL statements.

        :Arguments:
            **connection** (:obj:`NestedTransactionManager`): Manager owning the
            persistent connection used by this table.
        """
        if not isinstance(connection, NestedTransactionManager):
            raise TypeError("ProjectTable requires a NestedTransactionManager manager")
        if not self.name or not self.key or not self.record_name:
            raise TypeError(f"{self.__class__.__name__} must define a table name, key, and record name")
        for key, value in self.defaults.items():
            if value is None:
                raise ValueError(f"default value of None found for {key=}")

        self._connection = connection
        self._record_schema_version = -1
        self._table_info_sql = _TABLE_INFO_SQL.format(table=self.name)
        self._count_sql = _COUNT_SQL.format(table=self.name)
        self._contains_sql = _CONTAINS_SQL.format(table=self.name, key=self.key)
        self._delete_sql = _DELETE_SQL.format(table=self.name, key=self.key)
        self._max_key_sql = _MAX_KEY_SQL.format(table=self.name, key=self.key)
        self._change_key_sql = _CHANGE_KEY_SQL.format(table=self.name, key=self.key)
        self._non_existant_id_sql = _NON_EXISTANT_ID_SQL.format(table=self.name, key=self.key)

        self._refresh_record_type()

    def _refresh_record_type(self) -> None:
        """Refresh the generated record type after a schema change."""
        schema_version = self._connection._connection.execute(_SCHEMA_VERSION_SQL).fetchone()[0]
        if schema_version == self._record_schema_version:
            return
        self.record_type = guess_record_type(self._connection, self.name, self.record_name)
        self._record_fields = tuple(field.name for field in dataclass_fields(self.record_type))

        # FIXME: Can't use self._select_column(column) because sqlite allows double quoted identifiers to be string
        # literals when the identifier doesn't exist and a string literal is allowed. In >=3.12 we can use
        # connection.setconfig to disallow this.
        # https://sqlite.org/quirks.html#double_quoted_string_literals_are_accepted
        record_columns = ",".join(column for column in self._record_fields)
        self._select_all_sql = _SELECT_SQL.format(table=self.name, columns=record_columns)
        self._select_one_sql = _SELECT_ONE_SQL.format(table=self.name, key=self.key, columns=record_columns)
        self._record_schema_version = schema_version

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the current writable table columns, including user fields."""
        rows = self._connection._connection.execute(self._table_info_sql).fetchall()
        return tuple(row[1] for row in rows if row[1] != "ogc_fid")

    @property
    def fields(self) -> FieldEditor:
        """Return the metadata editor for this table's fields."""
        return FieldEditor(self._connection, self.name)

    @property
    def data(self) -> pd.DataFrame:
        """Return all table data."""
        return pd.read_sql(self._select_all_sql, self._connection._connection)

    def get(self, key: Any, column: str | None = None, default: Any = MISSING) -> Any:
        """Return one record identified by ``key``.

        :Arguments:
            **key** (:obj:`Any`): Value of the table's identifying column.

            **column**

            **default** (:obj:`Any`): Return value if no record matches the key.

        :Returns:
            **record** (:obj:`Any`): Record for the row.
        """
        if column is None:
            self._refresh_record_type()
            row = self._connection._connection.execute(self._select_one_sql, [key]).fetchone()
        else:
            # FIXME: See note at top for as to why we don't use self._select_column(column) here
            row = self._connection._connection.execute(
                _SELECT_ONE_SQL.format(table=self.name, key=self.key, columns=column),
                [key],
            ).fetchone()

        if row is None:
            if default is MISSING:
                raise self._missing_record(key)
            else:
                return default
        elif column is not None:
            return self._record_value(column=column, value=row[0])
        else:
            return self._build_record(row)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all records."""
        self._refresh_record_type()
        rows = self._connection._connection.execute(self._select_all_sql).fetchall()
        return iter(self._build_record(row) for row in rows)

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        return self._connection._connection.execute(self._count_sql).fetchone()[0]

    def __contains__(self, key: Any) -> bool:
        """Return whether a row with ``key`` exists."""
        return self._connection._connection.execute(self._contains_sql, [key]).fetchone() is not None

    def insert(self, **values: Any) -> Any:
        """Insert one record and return its explicit or generated key.

        :Arguments:
            **values** (:obj:`Any`): Column values for the new record. Omitted
            columns are left to SQLite defaults.

        :Returns:
            **key** (:obj:`Any`): Explicit or generated record key.
        """
        with self._connection as conn:
            row = self._prepare_insert(values)
            if row.get(self.key) is None:
                row[self.key] = self._next_key()

            sql = self._insert_statement(tuple(row))
            parameters = [self._database_value(value) for value in row.values()]
            conn.execute(sql, parameters)

        self._invalidate()
        return row[self.key]

    def update(self, key: Any, **values: Any) -> None:
        """Update the supplied fields of one record.

        :Arguments:
            **key** (:obj:`Any`): Key of the record to update.

            **values** (:obj:`Any`): Column values to write.
        """
        with self._connection as conn:
            sql = self._update_statement(tuple(values))
            parameters = [self._database_value(value) for value in values.values()]
            parameters.append(key)

            if conn.execute(sql, parameters).rowcount == 0:
                raise self._missing_record(key)

        self._invalidate()

    def delete(self, key: Any) -> None:
        """Delete one record identified by ``key``.

        :Arguments:
            **key** (:obj:`Any`): Key of the record to delete.
        """
        with self._connection as conn:
            cursor = conn.execute(self._delete_sql, [key])
            if cursor.rowcount == 0:
                raise self._missing_record(key)

        self._invalidate()

    def insert_from(self, frame: pd.DataFrame) -> list[Any]:
        """Atomically insert records identified by a DataFrame key column.

        If the key column is not provided, sequential keys starting from self._next() key is used.

        :Arguments:
            **frame** (:obj:`pandas.DataFrame`): Key and value columns to write.

        :Returns:
            **inserted rows keys** (:obj:`list[Any]`): The keys of the insert rows, explicit or generated.
        """
        generate_key = self.key not in frame.columns
        if generate_key and not self.has_numeric_key:
            raise ValueError("for non-numeric key tables, the key must be provided")

        with self._connection.transaction() as conn:
            if generate_key:
                next_key = self._next_key()
                frame = frame.assign(**{self.key: range(next_key, next_key + len(frame))})

            # Both functions can have the key in the columns
            columns = tuple(col for col in frame.columns if col != self.key) + (self.key,)
            rows = self._prepare_rows(frame, columns)
            conn.executemany(self._insert_statement(columns), rows)

        self._invalidate()
        return frame[self.key].to_list()

    def update_from(self, frame: pd.DataFrame, allow_missing: bool = False) -> int:
        """
        Atomically update records identified by a DataFrame key column.

        Keys which do not match existing entries will raise a ValueError unless ``allow_missing`` is True.

        :Arguments:
            **frame** (:obj:`pandas.DataFrame`): Key and value columns to write.

            **allow_missing** (:obj:`bool`, *Optional*): Allow missing keys.

        :Returns:
            **updated rows** (:obj:`int`): Number of submitted rows.
        """
        if not allow_missing:
            missing_rows = self.find_missing(frame[self.key].to_list(), fetch_limit=10 + 1)

            if missing_rows:
                raise ValueError(
                    f"update contained keys which do not exist: {_truncate_list_to_str(missing_rows, limit=10)}"
                )

        with self._connection.transaction() as conn:
            # No key here, needs to be specially handled
            columns = tuple(col for col in frame.columns if col != self.key)
            rows = self._prepare_rows(frame, columns + (self.key,))  # But we still need it in the rows
            conn.executemany(self._update_statement(columns), rows)

        self._invalidate()
        return len(rows)

    def delete_from(self, keys: list[Any], allow_missing: bool = False) -> int:
        """
        Atomically delete records identified by a list of keys.

        Keys which do not match existing entries will raise a ValueError unless ``allow_missing`` is True.

        :Arguments:
            **frame** (:obj:`list[Any]`): Keys to delete.

            **allow_missing** (:obj:`bool`, *Optional*): Allow missing keys.

        :Returns:
            **deleted rows** (:obj:`int`): Number of submitted rows.
        """

        if not allow_missing:
            missing_rows = self.find_missing(keys, fetch_limit=10 + 1)

            if missing_rows:
                raise ValueError(
                    f"delete contained keys which do not exist: {_truncate_list_to_str(missing_rows, limit=10)}"
                )

        with self._connection.transaction() as conn:
            conn.executemany(self._delete_sql, keys)

        self._invalidate()
        return len(keys)

    def _build_record(self, row: tuple[Any, ...]) -> Any:
        """Convert one SQLite row into the table's record type."""
        values = []
        for column, value in zip(self._record_fields, row, strict=True):
            values.append(self._record_value(column, value))
        return self.record_type(*values)

    def _prepare_insert(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Layer supplied values over defaults."""
        row = dict(self.defaults)
        row.update((column, value) for column, value in values.items() if value is not None)
        return row

    def _prepare_rows(self, frame: pd.DataFrame, value_columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
        """Convert DataFrame records into SQLite parameter rows."""
        if self.key not in frame.columns:
            raise ValueError(f"table key ({self.key}) not found in dataframe columns ({frame.columns})")

        frame = frame[list(value_columns)].fillna(self.defaults)
        return list(frame.itertuples(index=False, name=None))

    def _insert_statement(self, columns: tuple[str, ...]) -> str:
        """Return the INSERT statement for a column shape."""
        return _format_insert(self.name, columns, self._geometry_placeholder)

    def _update_statement(self, columns: tuple[str, ...]) -> str:
        """Return the UPDATE statement for a column shape."""
        return _format_update(self.name, self.key, columns, self._geometry_placeholder)

    def _next_key(self) -> Any:
        """Return the next numeric key available in the table."""
        current = self._connection._connection.execute(self._max_key_sql).fetchone()[0]
        return 1 if current is None else current + 1

    def _change_key(self, key: Any, new_key: Any) -> None:
        """Change a record key for tables exposing a renumber operation."""
        with self._connection as conn:
            cursor = conn.execute(self._change_key_sql, [new_key, key])
            if cursor.rowcount == 0:
                raise self._missing_record(key)
        self._invalidate()

    def find_missing(self, keys: list[Any], fetch_limit: int = -1) -> list[Any]:
        """
        Find which keys are not used. Returns a subset of unused keys.

        :Arguments:
            **keys** (:obj:`list[Any]`): Keys to check usage of.

            **fetch_limit** (:obj:`int`, *Optional*): Limit returned keys to this number. Defaults to all.

        :Returns:
            **missing keys** (:obj:`list[Any]`): Return a subset of the input **keys** which are not used.
        """
        if not keys:
            return []

        values = ",".join("(?)" for _ in keys)
        sql = self._non_existant_id_sql.format(values=values)
        cursor = self._connection._connection.execute(sql, keys)

        if fetch_limit >= 0:
            missing_rows = cursor.fetchmany(size=fetch_limit)
        else:
            missing_rows = cursor.fetchall()

        return [row[0] for row in missing_rows]

    def _missing_record(self, key: Any) -> ValueError:
        """Build the consistent error used when a record does not exist."""
        return ValueError(f"{self.name} has no record with {self.key}={key!r}")

    def _invalidate(self) -> None:
        """Non-fallible hook for invalidating derived in-memory state."""
        return None

    @abstractmethod
    def _select_column(self, column: str) -> str:
        """Format a record field for a SELECT list."""

    @abstractmethod
    def _record_value(self, column: str, value: Any) -> Any:
        """Convert one SQLite value into its record representation."""

    @abstractmethod
    def _database_value(self, value: Any) -> Any:
        """Convert one Python value into its SQLite representation."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} table={self.name!r}>"


class NonSpatialProjectTable(ProjectTable):
    """Base class for project tables without a geometry column."""

    def _select_column(self, column: str) -> str:
        return escape_identifier(column)

    def _record_value(self, column: str, value: Any) -> Any:
        return value

    def _database_value(self, value: Any) -> Any:
        return value


class SpatialProjectTable(ProjectTable):
    """Base class for project tables with a SpatiaLite geometry column."""

    multi_part = False
    srid = 4326

    def __init__(self, connection: NestedTransactionManager) -> None:
        srid = int(self.srid)
        template = _MULTI_GEOMETRY_PLACEHOLDER if self.multi_part else _GEOMETRY_PLACEHOLDER
        self._geometry_placeholder = template.format(srid=srid)
        super().__init__(connection)
        self._extent_sql = _EXTENT_SQL.format(table=self.name)

    def extent(self) -> Polygon:
        """Return the bounding polygon for the table's geometry layer."""
        data = self._connection._connection.execute(self._extent_sql).fetchone()[0]
        return shapely.wkb.loads(data)

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Return all table data."""
        return gpd.GeoDataFrame.from_postgis(
            self._select_all_sql, self._connection._connection, geom_col="geometry", crs=self.srid
        )

    def _select_column(self, column: str) -> str:
        if column == "geometry":
            return _GEOMETRY_COLUMN
        return escape_identifier(column)

    def _record_value(self, column: str, value: Any) -> Any:
        if column == "geometry" and value is not None:
            return shapely.wkb.loads(bytes(value))
        return value

    def _database_value(self, value: Any) -> Any:
        return value.wkb if isinstance(value, BaseGeometry) else value

    def _prepare_rows(self, frame: pd.DataFrame, value_columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
        """Convert bulk geometry values to WKB for SQLite bindings."""
        if "geometry" in value_columns:
            frame = frame.to_wkb()
        return super()._prepare_rows(frame, value_columns)


def _truncate_list_to_str(list: list[Any], limit: int):
    sample = list[:limit]
    sample_text = str(sample)
    if len(list) >= limit:
        sample_text = sample_text[:-1] + ", ...]"

    return sample_text
