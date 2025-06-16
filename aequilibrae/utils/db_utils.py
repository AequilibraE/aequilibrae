import sqlite3
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Union

import pandas as pd
from aequilibrae import logger


class AequilibraEConnection(sqlite3.Connection):
    """
    This custom factory class intends to solve the issue of premature commits when trying to use manual transaction control.

    After ``manual_transaction`` is called, context manager enters and exits are tracked via their depth, the
    ``sqlite3.Connection`` is placed into manual transaction control and a transaction is started. If another
    transaction is already in progress an RuntimeError is raised.
    When exiting with depth == 0, the normal context manager enter and exit is called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__manual_transaction: bool = False
        self.__depth: int = 0
        self.__isolation_level = self.isolation_level

    def manual_transaction(self):
        if self.__manual_transaction:
            raise RuntimeError(
                "cannot start a manual transaction while another manual transaction is already in progress"
            )
        elif self.in_transaction:
            raise RuntimeError("cannot start a manual transaction while in another transaction")

        logger.debug("Manual transaction control enabled")
        self.__depth = 0
        self.__manual_transaction = True
        self.__isolation_level = self.isolation_level
        self.isolation_level = None
        self.execute("BEGIN")
        return self

    def __enter__(self):
        logger.debug(f"Called __enter__ with {self.__manual_transaction=}, {self.__depth=}")
        if self.__manual_transaction:
            self.__depth += 1

            return super().__enter__() if self.__depth == 1 else self
        else:
            return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"Called __exit__ with {self.__manual_transaction=}, {self.__depth=}")
        if self.__manual_transaction:
            self.__depth -= 1

            if self.__depth <= 0:
                self.__manual_transaction = False
                res = super().__exit__(exc_type, exc_value, traceback)
                self.isolation_level = self.__isolation_level
                return res
        else:
            return super().__exit__(exc_type, exc_value, traceback)


def list_tables_in_db(conn: Connection):
    sql = "SELECT name FROM sqlite_master WHERE type ='table'"
    table_list = sorted([x[0].lower() for x in conn.execute(sql).fetchall() if "idx_" not in x[0].lower()])
    return table_list


def safe_connect(filepath: PathLike, missing_ok=False):
    if Path(filepath).exists() or missing_ok or str(filepath) == ":memory:":
        return connect(filepath, factory=AequilibraEConnection)
    raise FileNotFoundError(f"Attempting to open non-existant SQLite database: {filepath}")


class commit_and_close:
    """A context manager for sqlite connections which closes and commits."""

    def __init__(self, db: Union[str, Path, Connection], commit: bool = True, missing_ok: bool = False, spatial=False):
        """
        :Arguments:

            **db** (:obj:`Union[str, Path, Connection]`): The database (filename or connection) to be managed

            **commit** (:obj:`bool`): Boolean indicating if a commit/rollback should be attempted on closing

            **missing_ok** (:obj:`bool`): Boolean indicating that the db is not expected to exist yet
        """
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
    """A context manager for sqlite connections (alias for `commit_and_close(db,commit=False))`."""
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
