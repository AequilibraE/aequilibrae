import contextlib
import logging
import sqlite3
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class AequilibraEConnection(sqlite3.Connection):
    """SQLite connection type used by AequilibraE connection factories.

    Transaction ownership belongs to :class:`NestedTransactions`; this class
    intentionally has no alternative/native transaction-depth mode.
    """


class NestedTransactions:
    """Own one SQLite connection and provide freely nestable transactions.

    The connection is configured for explicit transaction control and foreign
    key enforcement.  SQL execution is delegated, but connection finalization
    remains private to the owning :class:`ConnectionClosure`.
    """

    def __init__(self, connection: Connection, *, _configured: bool = False):
        if not isinstance(connection, Connection):
            raise TypeError("NestedTransactions requires a sqlite3.Connection")
        if connection.in_transaction:
            raise ValueError("connections must not already be in a transaction")
        if not _configured:
            connection.isolation_level = None
            _enable_foreign_keys(connection)

        self.__connection = connection
        self.__depth = 0
        self.__savepoint_id = 0
        self.__closed = False

    @property
    def depth(self) -> int:
        """Current transaction/savepoint nesting depth (read-only)."""
        return self.__depth

    @property
    def in_transaction(self) -> bool:
        """Whether this manager has an active scope or SQLite transaction."""
        return self.__depth > 0 or (not self.__closed and self.__connection.in_transaction)

    def transaction(self):
        """Return a fresh transaction context whose value is the owned connection."""
        return _Transaction(self)

    def _connection(self):
        """Return the owned raw SQLite connection (for transaction contexts)."""
        return self.__connection

    # Deliberately limited DB-API delegation.  In particular, commit, rollback,
    # close, executescript, and a raw connection attribute are not exposed.
    def execute(self, sql, parameters=()):
        return self.__connection.execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        return self.__connection.executemany(sql, seq_of_parameters)

    def cursor(self, *args, **kwargs):
        return self.__connection.cursor(*args, **kwargs)

    @property
    def total_changes(self):
        return self.__connection.total_changes

    def _enter(self):
        if self.__closed:
            raise sqlite3.ProgrammingError("transaction manager is closed")
        savepoint = None
        if self.__depth == 0:
            if self.__connection.in_transaction:
                raise RuntimeError("managed connection has an unowned transaction")
            self.__connection.execute("BEGIN")
        else:
            self.__savepoint_id += 1
            savepoint = f"aeq_nested_{self.__savepoint_id}"
            self.__connection.execute(f'SAVEPOINT "{savepoint}"')
        self.__depth += 1
        return savepoint

    def _exit(self, savepoint, exc_type, exc_value):
        # Decrement only after entry succeeded, but before finalization so an
        # ExitStack-propagated commit failure makes earlier managers roll back.
        self.__depth -= 1
        connection = self.__connection
        try:
            if savepoint is None:
                connection.execute("COMMIT" if exc_type is None else "ROLLBACK")
            elif exc_type is None:
                connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
            else:
                connection.execute(f'ROLLBACK TO SAVEPOINT "{savepoint}"')
                connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        except BaseException as finalization_error:
            cleanup_error = self._cleanup_failed_finalization(savepoint)
            if exc_value is not None:
                _attach_cleanup_failure(exc_value, finalization_error)
                if cleanup_error is not None:
                    _attach_cleanup_failure(exc_value, cleanup_error)
                return False
            if cleanup_error is not None:
                _attach_cleanup_failure(finalization_error, cleanup_error)
            raise
        return False

    def _cleanup_failed_finalization(self, savepoint):
        connection = self.__connection
        if not connection.in_transaction:
            return None
        try:
            if savepoint is None:
                connection.execute("ROLLBACK")
            else:
                connection.execute(f'ROLLBACK TO SAVEPOINT "{savepoint}"')
                connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        except BaseException as error:  # preserve the primary exception
            logger.exception("Failed to clean up a SQLite transaction", exc_info=error)
            return error
        return None

    def _ensure_idle(self):
        if self.__depth or (not self.__closed and self.__connection.in_transaction):
            raise RuntimeError("cannot destroy a connection while a transaction is active")

    def _close(self):
        if self.__closed:
            return
        self._ensure_idle()
        self.__connection.close()
        self.__closed = True


class _Transaction:
    def __init__(self, manager: NestedTransactions):
        self.__manager = manager
        self.__savepoint = None
        self.__entered = False

    def __enter__(self):
        if self.__entered:
            raise RuntimeError("a transaction context cannot be entered twice")
        self.__savepoint = self.__manager._enter()
        self.__entered = True
        return self.__manager._connection()

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.__entered:
            return False
        self.__entered = False
        return self.__manager._exit(self.__savepoint, exc_type, exc_value)


class ConnectionClosure:
    """Own a non-empty set of distinct, named SQLite connections.

    A closure transaction coordinates normal unwind/rollback with ``ExitStack``.
    SQLite cannot atomically commit several independent connections, so a late
    commit failure can leave a connection that unwound earlier committed.
    """

    def __init__(self, connections):
        connections = dict(connections)
        if not connections:
            raise ValueError("expected at least one named connection")
        if any(not isinstance(connection, Connection) for connection in connections.values()):
            raise TypeError("every named value must be a sqlite3.Connection")
        if len({id(connection) for connection in connections.values()}) != len(connections):
            raise ValueError("each name must refer to a different connection")
        if any(connection.in_transaction for connection in connections.values()):
            raise ValueError("connections must not already be in a transaction")

        original_levels = {name: connection.isolation_level for name, connection in connections.items()}
        configured = []
        try:
            for name, connection in connections.items():
                configured.append(name)
                connection.isolation_level = None
                _enable_foreign_keys(connection)
        except BaseException:
            # Configuration has not transferred ownership yet.
            for name in reversed(configured):
                connections[name].isolation_level = original_levels[name]
            raise

        self.__managers = {
            name: NestedTransactions(connection, _configured=True) for name, connection in connections.items()
        }
        self.__closed = False

    def __getitem__(self, name) -> NestedTransactions:
        return self.__managers[name]

    def __iter__(self):
        return iter(self.__managers)

    @contextlib.contextmanager
    def transaction(self):
        if self.__closed:
            raise RuntimeError("connection closure is closed")
        with contextlib.ExitStack() as stack:
            yield {name: stack.enter_context(manager.transaction()) for name, manager in self.__managers.items()}

    def ensure_idle(self):
        for manager in self.__managers.values():
            manager._ensure_idle()

    def close(self):
        """Destroy all managers and their connections at the owner boundary."""
        if self.__closed:
            return
        self.ensure_idle()
        errors = []
        for manager in reversed(tuple(self.__managers.values())):
            try:
                manager._close()
            except BaseException as error:
                errors.append(error)
        self.__closed = True
        if errors:
            raise errors[0]


def _enable_foreign_keys(connection: Connection):
    connection.execute("PRAGMA foreign_keys = ON")
    row = connection.execute("PRAGMA foreign_keys").fetchone()
    if row is None or row[0] != 1:
        raise RuntimeError("could not enable SQLite foreign-key enforcement")


def _attach_cleanup_failure(primary: BaseException, cleanup: BaseException):
    note = f"Additional transaction cleanup failure: {cleanup!r}"
    if hasattr(primary, "add_note"):
        primary.add_note(note)
    else:  # pragma: no cover - Python versions without exception notes
        logger.error(note)


def list_tables_in_db(conn):
    sql = "SELECT name FROM sqlite_master WHERE type ='table'"
    table_list = sorted([x[0].lower() for x in conn.execute(sql).fetchall() if "idx_" not in x[0].lower()])
    return table_list


def safe_connect(filepath: PathLike, missing_ok=False):
    """Low-level connection factory; callers must immediately install an owner."""
    if Path(filepath).exists() or missing_ok or str(filepath) == ":memory:":
        connection = connect(filepath, factory=AequilibraEConnection)
        _enable_foreign_keys(connection)
        return connection
    raise FileNotFoundError(f"Attempting to open non-existent SQLite database: {filepath}")


class commit_and_close:
    """Legacy context manager pending conversion to an owning closure."""

    def __init__(self, db: Union[str, Path, Connection], commit: bool = True, missing_ok: bool = False, spatial=False):
        from aequilibrae.utils.spatialite_utils import connect_spatialite, load_spatialite_extension

        if spatial:
            if isinstance(db, Connection):
                load_spatialite_extension(db)
                self.conn = db
            elif not isinstance(db, (str, PathLike)):
                raise Exception("You must provide a database path to connect to spatialite")
            else:
                self.conn = connect_spatialite(db, missing_ok)
        elif isinstance(db, (str, PathLike)):
            self.conn = safe_connect(db, missing_ok)
        else:
            self.conn = db
        self.commit = commit

    def __enter__(self):
        return self.conn

    def __exit__(self, err_typ, err_value, traceback):
        if self.commit:
            if err_typ is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        self.conn.close()


def read_and_close(filepath, spatial=False):
    return commit_and_close(filepath, commit=False, spatial=spatial)


def read_sql(sql, filepath, **kwargs):
    with read_and_close(filepath) as conn:
        return pd.read_sql(sql, conn, **kwargs)


def has_table(conn, table_name):
    sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name like '{table_name}';"
    return len(conn.execute(sql).fetchall()) > 0


@dataclass
class ColumnDef:
    idx: int
    name: str
    type: str
    not_null: bool
    default: str
    is_pk: bool


def get_schema(conn, table_name):
    rv = [ColumnDef(*e) for e in conn.execute(f"PRAGMA table_info({table_name});").fetchall()]
    return {e.name: e for e in rv}


def list_columns(conn, table_name):
    return list(get_schema(conn, table_name).keys())


def has_column(conn, table_name, col_name):
    return col_name in get_schema(conn, table_name)


def add_column_unless_exists(conn, table_name, col_name, col_type, constraints=None):
    if not has_column(conn, table_name, col_name):
        add_column(conn, table_name, col_name, col_type, constraints)


def add_column(conn, table_name, col_name, col_type, constraints=None):
    sql = f"ALTER TABLE {table_name} ADD {col_name} {col_type} {constraints};"
    conn.execute(sql)
