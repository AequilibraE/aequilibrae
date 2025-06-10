import sqlite3
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Union

import pandas as pd


def list_tables_in_db(conn: Connection):
    sql = "SELECT name FROM sqlite_master WHERE type ='table'"
    table_list = sorted([x[0].lower() for x in conn.execute(sql).fetchall() if "idx_" not in x[0].lower()])
    return table_list


def safe_connect(filepath: PathLike, missing_ok=False):
    if Path(filepath).exists() or missing_ok or str(filepath) == ":memory:":
        return connect(filepath)
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
