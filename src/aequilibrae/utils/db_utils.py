"""SQLite connection ownership and small database helpers."""

import contextlib
import sqlite3
from collections.abc import Callable, Generator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any

import pandas as pd


class AequilibraEConnection(sqlite3.Connection):
    """SQLite connection type used by AequilibraE connection factories."""


class NestedTransactionManager:
    """Own one SQLite connection and provide nested transaction contexts."""

    def __init__(self, connection: Connection, *, configured: bool = False) -> None:
        if not isinstance(connection, Connection):
            raise TypeError("NestedTransactionManager requires a sqlite3.Connection")
        if connection.in_transaction:
            raise ValueError("connections must not already be in a transaction")
        if not configured:
            connection.isolation_level = None
            _enable_foreign_keys(connection)
        self.__connection = connection
        self.__depth = 0
        self.__savepoint_id = 0

    @property
    def connection(self) -> Connection:
        """The SQLite connection owned by this manager.

        Gateway code should normally obtain this connection from
        :meth:`transaction`; this property is for read-only integrations such
        as pandas that cannot consume a transaction context.
        """
        return self.__connection

    @property
    def depth(self) -> int:
        """Current transaction/savepoint nesting depth."""
        return self.__depth

    @property
    def in_transaction(self) -> bool:
        """Whether an owned transaction scope is currently active."""
        return self.__depth > 0 or self.__connection.in_transaction

    def transaction(self) -> "_TransactionContext":
        """Create a fresh context that yields the persistent connection."""
        return _TransactionContext(self)

    def _enter(self) -> str | None:
        savepoint: str | None = None
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

    def _exit(self, savepoint: str | None, exc_type: type[BaseException] | None) -> bool:
        self.__depth -= 1
        if savepoint is None:
            if exc_type is None:
                self.__connection.commit()
            else:
                self.__connection.rollback()
        elif exc_type is None:
            self.__connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        else:
            self.__connection.execute(f'ROLLBACK TO SAVEPOINT "{savepoint}"')
            self.__connection.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        return False

    def close(self) -> None:
        """Close the owned SQLite connection."""
        self.__connection.close()


class _TransactionContext:
    """A single-use transaction context created by a manager."""

    def __init__(self, manager: NestedTransactionManager) -> None:
        self.__manager = manager
        self.__savepoint: str | None = None

    def __enter__(self) -> Connection:
        self.__savepoint = self.__manager._enter()
        return self.__manager.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        return self.__manager._exit(self.__savepoint, exc_type)


class ConnectionClosure:
    """Own the project connection and optional results/transit connections."""

    def __init__(
        self,
        db_connection: Connection,
        results_connection: Connection | None = None,
        transit_connection: Connection | None = None,
    ) -> None:
        connections = (db_connection, results_connection, transit_connection)
        present_connections = tuple(connection for connection in connections if connection is not None)
        if not all(isinstance(connection, Connection) for connection in present_connections):
            raise TypeError("connection slots must be sqlite3.Connection instances or None")
        if len({id(connection) for connection in present_connections}) != len(present_connections):
            raise ValueError("each connection slot must refer to a different connection")
        if any(connection.in_transaction for connection in present_connections):
            raise ValueError("connections must not already be in a transaction")

        for connection in present_connections:
            connection.isolation_level = None
            _enable_foreign_keys(connection)

        self.__db_connection = NestedTransactionManager(db_connection, configured=True)
        self.__results_connection = (
            NestedTransactionManager(results_connection, configured=True) if results_connection is not None else None
        )
        self.__transit_connection = (
            NestedTransactionManager(transit_connection, configured=True) if transit_connection is not None else None
        )

    @classmethod
    def open(
        cls,
        db_opener: Callable[[], Connection],
        results_opener: Callable[[], Connection] | None = None,
        transit_opener: Callable[[], Connection] | None = None,
    ) -> "ConnectionClosure":
        """Open the known database slots, closing already-opened values on error."""
        opened_connections: list[Connection] = []
        try:
            db_connection = db_opener()
            opened_connections.append(db_connection)
            results_connection = results_opener() if results_opener is not None else None
            if results_connection is not None:
                opened_connections.append(results_connection)
            transit_connection = transit_opener() if transit_opener is not None else None
            if transit_connection is not None:
                opened_connections.append(transit_connection)
            closure = cls(db_connection, results_connection, transit_connection)
            opened_connections.clear()
            return closure
        except BaseException:
            for connection in opened_connections:
                connection.close()
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


def safe_connect(filepath: PathLike[str] | str, missing_ok: bool = False) -> Connection:
    """Open a non-spatial SQLite database without silently creating it."""
    if Path(filepath).exists() or missing_ok or str(filepath) == ":memory:":
        connection = connect(filepath, factory=AequilibraEConnection)
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
