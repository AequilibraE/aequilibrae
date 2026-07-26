"""One gateway for reading and writing the tables of an AequilibraE project.

``ProjectTable`` replaces the old ``SafeClass``/``TableLoader``/``BasicTable``/
``DataLoader`` stack and the per-row record classes built on them. It is a thin
table gateway, deliberately not an ORM:

- Reads are always fresh from the database. ``get()`` returns an immutable
  record (a frozen dataclass generated from the live table schema) — a
  throwaway snapshot, not a live object.
- Writes are explicit: you name the row and the columns. Single-row writes
  execute immediately; bulk writes go through ``batch()``, which builds an
  ordered script and flushes it through ``executemany`` in one transaction.
- Geometry is transparent: Shapely objects in, Shapely objects out. All
  WKB/SRID handling lives here.
"""

import logging
from contextlib import nullcontext
from dataclasses import make_dataclass
from itertools import count, groupby
from operator import itemgetter

import shapely.wkb
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.utils.get_table import get_geo_table

logger = logging.getLogger(__name__)


class ProjectTable:
    """Gateway to one table of the project database.

    .. code-block:: python

        >>> links = project.network.links

        # reading
        >>> gdf = links.data                # the whole table as a (Geo)DataFrame
        >>> link = links.get(3)             # one row, as an immutable record
        >>> link.modes
        'cMT'

        # writing one row
        >>> links.update(3, lanes_ab=4, name="Main St")

        # writing many rows
        >>> with links.batch() as b:
        ...     for link_id in (5, 7, 11):
        ...         b.update(link_id, speed_ab=90.0)
    """

    #: Name of the table in the database
    name = ""
    #: Primary key column
    key = ""
    #: Columns maintained by AequilibraE (e.g. by triggers) that cannot be written
    protected = frozenset()
    #: Whether the table has geometry and requires a spatialite connection
    spatial = False
    #: Promote written geometries to multi-part (e.g. zones)
    multi_part = False
    #: Values applied to inserts underneath the caller-supplied ones
    defaults = {}
    #: Name given to this table's record class
    record_name = ""

    srid = 4326

    def __init__(self, project):
        self.project = project
        self._record_cache = (None, None)

    # ------------------------------------------------------------------ schema

    @property
    def columns(self) -> tuple:
        """The table's columns, read from the database on demand (never stale)"""
        with self._read_ctx(None) as conn:
            return self._columns(conn)

    @property
    def fields(self) -> FieldEditor:
        """Returns a FieldEditor instance to edit this table's fields and their metadata"""
        return FieldEditor(self.project, self.name)

    # ----------------------------------------------------------------- reading

    @property
    def data(self):
        """The entire table as a DataFrame (GeoDataFrame for spatial tables)

        This is the starting point for bulk edits: modify the frame in pandas
        and push it back with ``update_from()``.
        """
        with self._read_ctx(None) as conn:
            return get_geo_table(self.name, conn)

    def get(self, key, conn=None):
        """Returns the record with the given key as an immutable snapshot

        Raises ``ValueError`` if the record does not exist. To change the
        record, write through ``update()``/``batch()``.
        """
        with self._read_ctx(conn) as connection:
            cols = self._columns(connection)
            sql = f'{self._select_sql(cols)} WHERE "{self.key}"=?'
            row = connection.execute(sql, [key]).fetchone()
        if row is None:
            raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        return self._build_record(cols, row)

    def __iter__(self):
        with self._read_ctx(None) as conn:
            cols = self._columns(conn)
            rows = conn.execute(self._select_sql(cols)).fetchall()
        return iter([self._build_record(cols, row) for row in rows])

    def __len__(self) -> int:
        with self._read_ctx(None) as conn:
            return conn.execute(f'SELECT COUNT(*) FROM "{self.name}"').fetchone()[0]

    def __contains__(self, key) -> bool:
        with self._read_ctx(None) as conn:
            sql = f'SELECT 1 FROM "{self.name}" WHERE "{self.key}"=? LIMIT 1'
            return conn.execute(sql, [key]).fetchone() is not None

    def extent(self) -> Polygon:
        """Queries the extent of the layer included in the model

        :Returns:
            **model extent** (:obj:`Polygon`): Shapely polygon with the bounding box of the layer.
        """
        if not self.spatial:
            raise TypeError(f"{self.name} is not a spatial table")
        with self._read_ctx(None) as conn:
            data = conn.execute(f'SELECT ST_AsBinary(GetLayerExtent("{self.name}"))').fetchone()[0]
        return shapely.wkb.loads(data)

    # ------------------------------------------------------------ writing: one

    def insert(self, conn=None, **values):
        """Inserts one record, returning its key

        Table defaults are applied underneath the given values. For tables
        with a numeric key, the key may be omitted and is assigned as max+1.
        """
        with self._write_ctx(conn) as connection:
            row = self._insert_row(values, self._columns(connection))
            if row.get(self.key) is None:
                row[self.key] = self._next_key(connection)
            connection.execute(*self._insert_sql(row))
        self._after_write()
        return row[self.key]

    def update(self, key, conn=None, **values):
        """Writes the given columns of one record — exactly those, nothing else

        Raises ``ValueError`` if the record does not exist.
        """
        with self._write_ctx(conn) as connection:
            values = self._checked(values, self._columns(connection), updating=True)
            sql, params = self._update_sql(values)
            if connection.execute(sql, [*params, key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._after_write()

    def delete(self, key, conn=None):
        """Removes one record from the table

        Raises ``ValueError`` if the record does not exist.
        """
        with self._write_ctx(conn) as connection:
            sql = f'DELETE FROM "{self.name}" WHERE "{self.key}"=?'
            if connection.execute(sql, [key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._after_write()

    # ----------------------------------------------------------- writing: many

    def batch(self, conn=None) -> "TableBatch":
        """Returns a batch for writing many records in one transaction

        The batch is an ordered script: queued inserts/updates/deletes execute
        in program order, with adjacent identical statements collapsed into a
        single ``executemany``. Use as a context manager — the script flushes
        on clean exit and is discarded if the block raises.
        """
        with self._read_ctx(conn) as connection:
            cols = self._columns(connection)
        return TableBatch(self, cols, conn)

    def update_from(self, df, conn=None) -> int:
        """Updates records from a DataFrame holding the key column plus value columns

        All rows must provide the same columns (that is what a DataFrame is);
        the write is a single ``executemany``. Returns the number of rows written.
        """
        frame = df.reset_index() if self.key in df.index.names else df
        if self.key not in frame.columns:
            raise ValueError(f"The DataFrame needs a '{self.key}' column (or index) to identify records")

        value_cols = [c for c in frame.columns if c != self.key]
        if not value_cols:
            raise ValueError("Nothing to update: the DataFrame only contains the key")

        with self._write_ctx(conn) as connection:
            self._check_columns(value_cols, self._columns(connection), updating=True)
            sql = f'{self._set_sql(value_cols)} WHERE "{self.key}"=?'
            rows = [[*(_db_value(row[c]) for c in value_cols), row[self.key]] for row in frame.to_dict("records")]
            connection.executemany(sql, rows)
        self._after_write()
        return len(rows)

    def insert_from(self, df, conn=None) -> list:
        """Inserts every row of a DataFrame, returning the new keys

        Missing numeric keys are assigned sequentially from max+1.
        """
        frame = df.reset_index() if self.key in df.index.names else df
        with self.batch(conn) as batch:
            return [batch.insert(**row) for row in frame.to_dict("records")]

    # --------------------------------------------------------------- internals

    def _connect(self):
        return self.project.db_connection_spatial if self.spatial else self.project.db_connection

    def _read_ctx(self, conn):
        """Borrowed connections are used as-is (no transaction side effects)"""
        return nullcontext(conn) if conn is not None else self._connect()

    def _write_ctx(self, conn):
        """Borrowed connections commit via sqlite's own transaction context"""
        return conn if conn is not None else self._connect()

    def _columns(self, conn) -> tuple:
        dt = conn.execute(f'pragma table_info("{self.name}")').fetchall()
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
        """Validated values over the table defaults; a ``None`` value means "not provided" """
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
        return {k: self._check_value(k, v) for k, v in values.items()}

    def _check_value(self, column, value):
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

    def _next_key(self, conn):
        current = conn.execute(f'SELECT MAX("{self.key}") FROM "{self.name}"').fetchone()[0]
        if current is None:
            return 1
        if not isinstance(current, int):
            raise ValueError(f"{self.name}.{self.key} is not numeric, so the {self.key} must be given explicitly")
        return current + 1

    def _change_key(self, key, new_key, conn=None):
        with self._write_ctx(conn) as connection:
            sql = f'UPDATE "{self.name}" SET "{self.key}"=? WHERE "{self.key}"=?'
            if connection.execute(sql, [new_key, key]).rowcount == 0:
                raise ValueError(f"{self.name} has no record with {self.key}={key!r}")
        self._after_write()

    def _after_write(self):
        """Hook for subclasses holding derived state (e.g. a spatial index)"""

    def __copy__(self):
        raise TypeError(f"{self.__class__.__name__} objects cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise TypeError(f"{self.__class__.__name__} objects cannot be copied")

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.name!r}>"


class TableBatch:
    """An ordered script of inserts/updates/deletes against one table.

    Statements execute in the order they were queued, inside a single
    transaction; adjacent statements with identical SQL are collapsed into one
    ``executemany`` call. Values are validated when queued, so mistakes fail
    fast, before anything touches the database.
    """

    def __init__(self, table: ProjectTable, cols, conn=None):
        self._table = table
        self._cols = cols
        self._conn = conn
        self._script = []
        self._keys = None

    def insert(self, **values):
        """Queues one insert, returning the key the record will get"""
        table = self._table
        row = table._insert_row(values, self._cols)
        if row.get(table.key) is None:
            row[table.key] = self._assign_key()
        self._script.append(table._insert_sql(row))
        return row[table.key]

    def update(self, key, **values):
        """Queues an update of the given columns of one record"""
        sql, params = self._table._update_sql(self._table._checked(values, self._cols, updating=True))
        self._script.append((sql, [*params, key]))

    def delete(self, key):
        """Queues the removal of one record"""
        self._script.append((f'DELETE FROM "{self._table.name}" WHERE "{self._table.key}"=?', [key]))

    def flush(self):
        """Executes the queued script in one transaction and clears it"""
        if not self._script:
            return
        with self._table._write_ctx(self._conn) as conn:
            for sql, rows in groupby(self._script, key=itemgetter(0)):
                params = [p for _, p in rows]
                if len(params) == 1:
                    conn.execute(sql, params[0])
                else:
                    conn.executemany(sql, params)
        self._script.clear()
        self._table._after_write()

    def _assign_key(self):
        if self._keys is None:
            with self._table._read_ctx(self._conn) as conn:
                self._keys = count(self._table._next_key(conn))
        return next(self._keys)

    def __len__(self) -> int:
        return len(self._script)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.flush()
        return False


def _db_value(value):
    """Shapely geometries travel to the database as WKB; everything else as-is"""
    return value.wkb if isinstance(value, BaseGeometry) else value
