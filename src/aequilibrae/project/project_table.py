"""Persistent-connection table gateways for AequilibraE projects."""

import logging
from dataclasses import make_dataclass
from itertools import count

import shapely.wkb
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.utils.db_utils import NestedTransactions
from aequilibrae.utils.get_table import get_geo_table

logger = logging.getLogger(__name__)


class ProjectTable:
    """Thin gateway for one database table.

    Reads use the injected persistent manager. Every mutation owns a nested
    transaction, which is a top-level transaction for standalone calls and a
    savepoint when enclosed by ``Project.transaction()``.
    """

    name = ""
    key = ""
    protected = frozenset()
    spatial = False
    multi_part = False
    defaults = {}
    record_name = ""
    srid = 4326

    def __init__(self, transactions: NestedTransactions):
        if not isinstance(transactions, NestedTransactions):
            raise TypeError("ProjectTable requires a NestedTransactions manager")
        self._transactions = transactions
        self._record_cache = (None, None)

    @property
    def columns(self) -> tuple:
        return self._columns()

    @property
    def fields(self) -> FieldEditor:
        return FieldEditor(self._transactions, self.name)

    @property
    def data(self):
        """Return the table indexed exclusively by its named record key."""
        frame = get_geo_table(self.name, self._transactions)
        return frame.set_index(self.key, drop=True)

    def get(self, key):
        cols = self._columns()
        sql = f'{self._select_sql(cols)} WHERE "{self.key}"=?'
        row = self._transactions.execute(sql, [key]).fetchone()
        if row is None:
            raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        return self._build_record(cols, row)

    def __iter__(self):
        cols = self._columns()
        rows = self._transactions.execute(self._select_sql(cols)).fetchall()
        return iter([self._build_record(cols, row) for row in rows])

    def __len__(self) -> int:
        return self._transactions.execute(f'SELECT COUNT(*) FROM "{self.name}"').fetchone()[0]

    def __contains__(self, key) -> bool:
        sql = f'SELECT 1 FROM "{self.name}" WHERE "{self.key}"=? LIMIT 1'
        return self._transactions.execute(sql, [key]).fetchone() is not None

    def extent(self) -> Polygon:
        if not self.spatial:
            raise TypeError(f"{self.name} is not a spatial table")
        data = self._transactions.execute(f'SELECT ST_AsBinary(GetLayerExtent("{self.name}"))').fetchone()[0]
        return shapely.wkb.loads(data)

    def insert(self, **values):
        with self._transactions.transaction():
            row = self._insert_row(values, self._columns())
            if row.get(self.key) is None:
                row[self.key] = self._next_key()
            self._transactions.execute(*self._insert_sql(row))
        self._invalidate()
        return row[self.key]

    def update(self, key, **values):
        with self._transactions.transaction():
            values = self._checked(values, self._columns(), updating=True)
            sql, params = self._update_sql(values)
            if self._transactions.execute(sql, [*params, key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._invalidate()

    def delete(self, key):
        with self._transactions.transaction():
            sql = f'DELETE FROM "{self.name}" WHERE "{self.key}"=?'
            if self._transactions.execute(sql, [key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._invalidate()

    def update_from(self, df) -> int:
        """Update rows identified only by a unique, non-missing named index."""
        if df.index.name != self.key:
            raise ValueError(f"The DataFrame index must be named '{self.key}' to identify records")
        if self.key in df.columns:
            raise ValueError(f"'{self.key}' cannot be both the update index and a value column")
        if not df.index.is_unique:
            raise ValueError("The update DataFrame index must be unique")
        if df.index.hasnans:
            raise ValueError("The update DataFrame index cannot contain missing values")
        value_cols = list(df.columns)
        if not value_cols:
            raise ValueError("Nothing to update: the DataFrame only contains the key index")

        with self._transactions.transaction():
            columns = self._columns()
            self._check_columns(value_cols, columns, updating=True)
            rows = []
            for key, values in df.iterrows():
                if key not in self:
                    raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
                checked = [self._validate_value(column, values[column]) for column in value_cols]
                rows.append([*(_db_value(value) for value in checked), key])
            sql = f'{self._set_sql(value_cols)} WHERE "{self.key}"=?'
            self._transactions.executemany(sql, rows)
        self._invalidate()
        return len(rows)

    def insert_from(self, df) -> list:
        """Atomically insert every DataFrame row and return their keys."""
        frame = df.reset_index() if self.key in df.index.names else df
        with self._transactions.transaction():
            columns = self._columns()
            keys = count(self._next_key()) if self.key not in frame.columns else None
            rows = []
            inserted_keys = []
            for values in frame.to_dict("records"):
                row = self._insert_row(values, columns)
                if row.get(self.key) is None:
                    if keys is None:
                        keys = count(self._next_key())
                    row[self.key] = next(keys)
                inserted_keys.append(row[self.key])
                rows.append(self._insert_sql(row))
            for sql, parameters in rows:
                self._transactions.execute(sql, parameters)
        self._invalidate()
        return inserted_keys

    def _columns(self) -> tuple:
        dt = self._transactions.execute(f'pragma table_info("{self.name}")').fetchall()
        return tuple(x[1] for x in dt if x[1] != "ogc_fid")

    def _select_sql(self, cols) -> str:
        keys = ",".join(f'ST_AsBinary("{c}")' if c == "geometry" else f'"{c}"' for c in cols)
        return f'SELECT {keys} FROM "{self.name}"'

    def _build_record(self, cols, row):
        cached_cols, record_type = self._record_cache
        if cached_cols != cols:
            record_type = make_dataclass(self.record_name or f"{self.name}_record", cols, frozen=True)
            self._record_cache = (cols, record_type)
        values = (
            shapely.wkb.loads(bytes(v)) if c == "geometry" and v is not None else v
            for c, v in zip(cols, row, strict=True)
        )
        return record_type(*values)

    def _insert_row(self, values: dict, cols) -> dict:
        row = {k: v for k, v in self._checked(values, cols, updating=False).items() if v is not None}
        return {**self.defaults, **row}

    def _check_columns(self, requested, cols, updating: bool):
        unknown = [c for c in requested if c not in cols]
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)}: not fields of the {self.name} table. Fields are: {', '.join(cols)}"
            )
        if updating and self.key in requested:
            raise ValueError(f'"{self.key}" identifies the record and cannot be updated. See renumber(), if available')
        blocked = [c for c in requested if c in self.protected]
        if blocked:
            raise ValueError(f"{', '.join(blocked)}: maintained by AequilibraE and cannot be written directly")

    def _checked(self, values: dict, cols, updating: bool) -> dict:
        self._check_columns(values.keys(), cols, updating)
        return {k: self._validate_value(k, v) for k, v in values.items()}

    def _validate_value(self, column, value):
        if column == "geometry":
            if not isinstance(value, BaseGeometry):
                raise TypeError("geometry must be a Shapely geometry object")
            return value
        check = getattr(self, f"_check_{column}", None)
        return check(value) if check is not None else value

    def _geom_expr(self) -> str:
        expr = f"GeomFromWKB(?, {int(self.srid)})"
        return f"ST_Multi({expr})" if self.multi_part else expr

    def _placeholder(self, column) -> str:
        return self._geom_expr() if column == "geometry" else "?"

    def _insert_sql(self, row: dict):
        row = {k: v for k, v in row.items() if v is not None}
        cols = ",".join(f'"{k}"' for k in row)
        marks = ",".join(self._placeholder(k) for k in row)
        sql = f'INSERT INTO "{self.name}" ({cols}) VALUES ({marks})'
        return sql, [_db_value(v) for v in row.values()]

    def _set_sql(self, columns) -> str:
        sets = ",".join(f'"{c}"={self._placeholder(c)}' for c in columns)
        return f'UPDATE "{self.name}" SET {sets}'

    def _update_sql(self, values: dict):
        if not values:
            raise ValueError("Nothing to update: no values were given")
        sql = f'{self._set_sql(values.keys())} WHERE "{self.key}"=?'
        return sql, [_db_value(v) for v in values.values()]

    def _next_key(self):
        current = self._transactions.execute(f'SELECT MAX("{self.key}") FROM "{self.name}"').fetchone()[0]
        if current is None:
            return 1
        if not isinstance(current, int):
            raise ValueError(f"{self.name}.{self.key} is not numeric, so the {self.key} must be given explicitly")
        return current + 1

    def _change_key(self, key, new_key):
        with self._transactions.transaction():
            sql = f'UPDATE "{self.name}" SET "{self.key}"=? WHERE "{self.key}"=?'
            if self._transactions.execute(sql, [new_key, key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._invalidate()

    def _invalidate(self):
        """Non-fallible hook for invalidating derived in-memory state."""

    def __copy__(self):
        raise TypeError(f"{self.__class__.__name__} objects cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise TypeError(f"{self.__class__.__name__} objects cannot be copied")

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.name!r}>"


def _db_value(value):
    return value.wkb if isinstance(value, BaseGeometry) else value
