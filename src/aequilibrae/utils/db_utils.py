"""SQLite connection ownership and small database helpers."""

import contextlib
import logging
import shutil
from collections.abc import Generator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any

import pandas as pd
from pandas.api import types as pd_types

logger = logging.getLogger(__name__)


class _AequilibraEConnection(Connection):
    """SQLite connection type used by AequilibraEs NestedTransactionManager."""

    # FIXME: This is a big hack, in order to still allow the NestedTransactionManager to work with pandas we yield the
    # raw connection in the context manager, however, we can't stop pandas or anyone else from calling .commit and
    # breaking our transaction control, so we make those operations no-ops. I really don't like this but I can't think
    # of a better way without bringing in something like SQLAlchemy which pandas supports explicitly.

    def commit(self, *_args, **_kwargs):
        logger.debug(
            f"commit was called an {self.__class__.__name__}, however this function is disabled in favour "
            "of the NestedTransactionManager. Call _commit if you truly must commit manually."
        )

    def rollback(self, *_args, **_kwargs):
        logger.debug(
            f"rollback was called an {self.__class__.__name__}, however this function is disabled in favour "
            "of the NestedTransactionManager. Call _rollback if you truly must rollback manually."
        )

    def __enter__(self):
        logger.debug(
            f"__enter__ was called an {self.__class__.__name__}, however the context manager on the connection "
            "is disabled in favour of the NestedTransactionManager."
        )
        return self

    def __exit__(self, *_args, **_kwargs):
        pass

    _commit = Connection.commit

    _rollback = Connection.rollback


class NestedTransactionManager:
    """Manage a SQLite connection and provide nested transaction contexts."""

    def __init__(
        self,
        path: PathLike[str] | str,
        *,
        spatial: bool = False,
        open_close: bool = False,
    ) -> None:
        if not isinstance(path, (str, PathLike)):
            raise TypeError("NestedTransactionManager requires a database path")
        elif path == ":memory:" and open_close:
            raise ValueError(
                "a memory only database with open_close enabled will be wiped each transaction, this is not allowed"
            )

        self.__path = path
        self.__spatial = spatial
        self.__open_close = open_close
        self.__connection: Connection | None = None
        self.__savepoint_id = 0
        self.__stack: list[str | None] = []

        if not self.__open_close:
            self.__connection = self.__open()

    @staticmethod
    def __configure(connection: Connection) -> None:
        """Configure a newly managed connection."""
        if not isinstance(connection, Connection):
            raise TypeError("expected a sqlite3.Connection")
        if connection.in_transaction:
            raise ValueError("connections must not already be in a transaction")
        connection.isolation_level = None
        _enable_foreign_keys(connection)

    def __open(self) -> Connection:
        """Open and configure a connection owned by this manager."""
        if self.__spatial:
            from aequilibrae.utils.spatialite_utils import connect_spatialite

            connection = connect_spatialite(self.__path, factory=_AequilibraEConnection)
        else:
            connection = safe_connect(self.__path, factory=_AequilibraEConnection)

        try:
            self.__configure(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    def __close(self) -> None:
        """Close and discard the current connection, if any."""
        connection = self.__connection
        if connection is None:
            return
        try:
            connection.close()
        finally:
            self.__connection = None

    @property
    def _connection(self) -> Connection:
        """The SQLite connection owned by this manager.

        Table code should normally obtain this connection from
        :meth:`transaction`; this property is for read-only integrations such
        as pandas that cannot consume a transaction context.
        """
        if self.__connection is None:
            raise RuntimeError("managed connection is closed outside a transaction")
        return self.__connection

    @property
    def depth(self) -> int:
        """Current transaction/savepoint nesting depth. Includes no-op transactions."""
        return len(self.__stack)

    @property
    def in_transaction(self) -> bool:
        """Whether an owned transaction scope is currently active."""
        return self.depth > 0 or (self.__connection is not None and self.__connection.in_transaction)

    @property
    def transaction_id(self) -> str:
        """
        Return the current savepoint ID.

        Raises a RuntimeError is not currently in a transaction
        """
        if not self.in_transaction:
            raise RuntimeError("not in a transaction")
        return next(transaction_id for transaction_id in reversed(self.__stack) if transaction_id is not None)

    def transaction(self) -> "_TransactionContext":
        """Create a fresh context that yields the managed connection."""
        return _TransactionContext(self)

    def __enter__(self) -> Connection:
        """
        When used as a context manager without a .transaction(), the enter and exit methods should no-op if there is
        an existing transaction, otherwise one is started.
        """
        if self.depth > 0:
            # There are things on the transaction stack, but .transaction() haven't been used, therefore this is a
            # noop-context
            self.__stack.append(None)
        else:
            # There's either nothing on the stack, thus we must start a new transaction
            self._enter_transaction()

        return self._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        savepoint = self.__stack[-1]
        if savepoint is None:
            # Then this is a noop-context
            self.__stack.pop()
            return False
        return self._exit_transaction(savepoint, exc_type)

    def _enter_transaction(self) -> str:
        opened_here = self.depth == 0 and self.__open_close
        if opened_here:
            self.__connection = self.__open()

        connection = self._connection
        try:
            if self.depth == 0 and connection.in_transaction:
                raise RuntimeError("managed connection has an unowned transaction")

            self.__savepoint_id += 1
            savepoint = f"aeq_nested_{self.__savepoint_id}"
            connection.execute(f'SAVEPOINT "{savepoint}"')
            self.__stack.append(savepoint)
            return savepoint
        except BaseException:
            if opened_here:
                self.__close()
            raise

    def _exit_transaction(self, savepoint: str | None, exc_type: type[BaseException] | None) -> bool:
        assert self.__stack.pop() == savepoint, "tried to exit a different transaction than was on top of the stack"
        connection = self._connection

        try:
            if exc_type is None:
                connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
            else:
                connection.execute(f'ROLLBACK TO SAVEPOINT "{savepoint}"')
                connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        finally:
            if self.depth == 0 and self.__open_close:
                self.__close()

        return False

    def close(self) -> None:
        """Close the owned SQLite connection."""
        if self.in_transaction:
            raise RuntimeError("cannot close while a transaction is active")
        self.__close()


class _TransactionContext:
    """A single-use transaction context created by a manager."""

    def __init__(self, manager: NestedTransactionManager) -> None:
        self.__manager = manager
        self.__savepoint: str | None = None

    def __enter__(self) -> Connection:
        self.__savepoint = self.__manager._enter_transaction()
        return self.__manager._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        return self.__manager._exit_transaction(self.__savepoint, exc_type)


class ConnectionClosure:
    """Own the project connection and optional results/transit connections."""

    def __init__(
        self,
        db_path: PathLike[str] | str,
        results_path: PathLike[str] | str | None = None,
        transit_path: PathLike[str] | str | None = None,
        *,
        open_close: bool = False,
    ) -> None:
        paths = (db_path, results_path, transit_path)
        present_paths = tuple(path for path in paths if path is not None)
        if not all(isinstance(path, (str, PathLike)) for path in present_paths):
            raise TypeError("database slots must be paths or None")

        managers: list[NestedTransactionManager] = []
        try:
            self.__db_connection = NestedTransactionManager(db_path, spatial=True, open_close=open_close)
            managers.append(self.__db_connection)

            self.__results_connection = (
                NestedTransactionManager(results_path, spatial=False, open_close=open_close)
                if results_path is not None
                else None
            )
            if self.__results_connection is not None:
                managers.append(self.__results_connection)

            self.__transit_connection = (
                NestedTransactionManager(transit_path, spatial=True, open_close=open_close)
                if transit_path is not None
                else None
            )
            if self.__transit_connection is not None:
                managers.append(self.__transit_connection)

        except BaseException:
            for manager in reversed(managers):
                manager.close()
            raise

    @property
    def db_connection(self) -> NestedTransactionManager:
        """The required project-database transaction manager."""
        return self.__db_connection

    @property
    def results_connection(self) -> NestedTransactionManager:
        """The results-database manager, when that database exists."""
        if self.__results_connection is None:
            raise RuntimeError("This scenario has no results database")
        return self.__results_connection

    @property
    def transit_connection(self) -> NestedTransactionManager:
        """The transit-database manager, when that database exists."""
        if self.__transit_connection is None:
            raise RuntimeError("This scenario has no transit database")
        return self.__transit_connection

    def create_results_connection(self, path: PathLike[str] | str) -> NestedTransactionManager:
        """Create and begin owning an empty results database."""
        if self.__results_connection is not None:
            raise RuntimeError("This scenario already has a results database")

        path = Path(path)
        if path.exists():
            raise FileExistsError(f"results database already exists: {path}")
        try:
            path.touch()
            self.__results_connection = NestedTransactionManager(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return self.__results_connection

    def create_transit_connection(self, path: PathLike[str] | str) -> NestedTransactionManager:
        """Create, initialise, and begin owning a transit database."""
        if self.__transit_connection is not None:
            raise RuntimeError("This scenario already has a transit database")

        from aequilibrae.project.project_creation import initialize_tables
        from aequilibrae.reference_files import spatialite_database

        path = Path(path)
        if path.exists():
            raise FileExistsError(f"transit database already exists: {path}")
        try:
            shutil.copyfile(spatialite_database, path)
            self.__transit_connection = NestedTransactionManager(path, spatial=True)
            initialize_tables("transit", conn=self.__transit_connection._connection)
        except BaseException:
            if self.__transit_connection is not None:
                self.__transit_connection.close()
                self.__transit_connection = None
            path.unlink(missing_ok=True)
            raise
        return self.__transit_connection

    @property
    def has_results_connection(self) -> bool:
        """Whether a results database is owned."""
        return self.__results_connection is not None

    @property
    def has_transit_connection(self) -> bool:
        """Whether a transit database is owned."""
        return self.__transit_connection is not None

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Enter transaction contexts for every existing database connection."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.__db_connection.transaction())
            if self.__results_connection is not None:
                stack.enter_context(self.__results_connection.transaction())
            if self.__transit_connection is not None:
                stack.enter_context(self.__transit_connection.transaction())
            yield

    def close(self) -> None:
        """Close every owned connection."""
        managers = (self.__transit_connection, self.__results_connection, self.__db_connection)
        for manager in managers:
            if manager is not None:
                manager.close()


def _enable_foreign_keys(connection: Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    row = connection.execute("PRAGMA foreign_keys").fetchone()
    if row is None or row[0] != 1:
        raise RuntimeError("could not enable SQLite foreign-key enforcement")


def list_tables_in_db(connection: Connection) -> list[str]:
    """List ordinary tables in a SQLite database."""
    sql = "SELECT name FROM sqlite_master WHERE type ='table'"
    return sorted(row[0].lower() for row in connection.execute(sql).fetchall() if "idx_" not in row[0].lower())


def safe_connect(filepath: PathLike[str] | str, missing_ok: bool = False, factory=Connection) -> Connection:
    """Open a non-spatial SQLite database without silently creating it."""
    if Path(filepath).exists() or missing_ok or str(filepath) == ":memory:":
        connection = connect(filepath, factory=factory)
        _enable_foreign_keys(connection)
        return connection
    raise FileNotFoundError(f"Attempting to open non-existent SQLite database: {filepath}")


class commit_and_close:
    """Legacy standalone connection context manager.

    New project code must use :class:`ConnectionClosure` and a
    :class:`NestedTransactionManager` instead.
    """

    def __init__(
        self,
        db: str | Path | Connection,
        commit: bool = True,
        missing_ok: bool = False,
        spatial: bool = False,
    ) -> None:
        from aequilibrae.utils.spatialite_utils import connect_spatialite, load_spatialite_extension

        if spatial:
            if isinstance(db, Connection):
                load_spatialite_extension(db)
                self.conn = db
            elif isinstance(db, (str, PathLike)):
                self.conn = connect_spatialite(db, missing_ok)
            else:
                raise TypeError("db must be a database path or sqlite3.Connection")
        elif isinstance(db, (str, PathLike)):
            self.conn = safe_connect(db, missing_ok)
        else:
            self.conn = db
        self.commit = commit

    def __enter__(self) -> Connection:
        return self.conn

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any) -> None:
        if self.commit:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        self.conn.close()


def read_and_close(filepath: PathLike[str] | str, spatial: bool = False) -> commit_and_close:
    """Return a legacy read-only standalone connection context."""
    return commit_and_close(filepath, commit=False, spatial=spatial)


def read_sql(sql: str, filepath: PathLike[str] | str, **kwargs: Any) -> pd.DataFrame:
    """Read a SQL query from a standalone SQLite database."""
    with read_and_close(filepath) as connection:
        return pd.read_sql(sql, connection, **kwargs)


def has_table(connection: Connection, table_name: str) -> bool:
    """Return whether a table with ``table_name`` exists."""
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name like ?"
    return connection.execute(sql, (table_name,)).fetchone() is not None


@dataclass
class ColumnDef:
    """SQLite column metadata."""

    idx: int
    name: str
    type: str
    not_null: bool
    default: str | None
    is_pk: bool


def get_schema(connection: Connection, table_name: str) -> dict[str, ColumnDef]:
    """Return schema metadata keyed by column name."""
    columns = [ColumnDef(*row) for row in connection.execute(f"PRAGMA table_info({table_name});").fetchall()]
    return {column.name: column for column in columns}


def list_columns(connection: Connection, table_name: str) -> list[str]:
    """Return a table's column names."""
    return list(get_schema(connection, table_name))


def has_column(connection: Connection, table_name: str, column_name: str) -> bool:
    """Return whether a table has ``column_name``."""
    return column_name in get_schema(connection, table_name)


def add_column_unless_exists(
    connection: Connection,
    table_name: str,
    column_name: str,
    column_type: str,
    constraints: str | None = None,
) -> None:
    """Add a column when it is not already present."""
    if not has_column(connection, table_name, column_name):
        add_column(connection, table_name, column_name, column_type, constraints)


def add_column(
    connection: Connection,
    table_name: str,
    column_name: str,
    column_type: str,
    constraints: str | None = None,
) -> None:
    """Add a SQLite column."""
    connection.execute(f"ALTER TABLE {table_name} ADD {column_name} {column_type} {constraints or ''};")


def escape_identifier(name) -> str:
    # See https://stackoverflow.com/a/6701665
    name = str(name).encode("utf-8", "strict").decode("utf-8")

    if not len(name):
        raise ValueError("identifier cannot be empty")

    if name.find("\x00") >= 0:
        raise ValueError("identifier cannot contain NULLs")

    return '"' + name.replace('"', '""') + '"'


def df_sqlite_types(frame: pd.DataFrame, overrides: dict) -> dict:
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

        elif pd_types.is_bool_dtype(frame[column].dtype) or pd_types.is_integer_dtype(frame[column].dtype):
            result[column] = "INTEGER"
        elif pd_types.is_float_dtype(frame[column].dtype):
            result[column] = "REAL"
        elif pd_types.is_object_dtype(frame[column].dtype) and all(
            isinstance(value, bytes) for value in frame[column].dropna()
        ):
            result[column] = "BLOB"
        else:
            result[column] = "TEXT"
    return result
