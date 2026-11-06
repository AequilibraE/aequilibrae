import logging
import sqlite3
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Optional

import openmatrix as omx

from aequilibrae.project.project_creation import add_triggers, initialize_tables

logger = logging.getLogger(__name__)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Returns if 'conn' has a table named 'name'"""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection] = None,
    results_conn: Optional[sqlite3.Connection] = None,
):
    if project_conn is None:
        raise RuntimeError("Network migration 011 requires a project_conn connection")

    if transit_conn is not None:
        status = (
            transit_conn.execute("SELECT status FROM migrations WHERE id=4").fetchone()
            if _table_exists(transit_conn, "migrations")
            else None
        )

        if status is None or status[0] not in {"APPLIED", "SKIPPED"}:
            raise RuntimeError(
                "Network migration 011 requires transit migration 004 to be APPLIED or SKIPPED before proceeding"
            )

    logger.info("Beginning migration to move the results table in project_database.sqlite to results_database.sqlite")

    # Need the path to the folder to do these matrix moves
    database_path = next(row[2] for row in project_conn.execute("PRAGMA database_list") if row[1] == "main")
    folder = Path(database_path).parent if database_path else None

    with TemporaryDirectory(dir=folder) as temporary:
        staged_files = stage_matrix_metadata(project_conn, folder, Path(temporary))
        move_results_table(project_conn, results_conn)

        # SQLite rollback cannot undo file replacements, so finish all staged edits and SQL work first.
        for copy, original in staged_files:
            copy.replace(original)


def stage_matrix_metadata(project_conn: sqlite3.Connection, folder: Optional[Path], temporary: Path):
    if not _table_exists(project_conn, "matrices"):
        return []

    columns = ("procedure", "procedure_id", "timestamp", "description")
    rows = project_conn.execute(f"SELECT file_name, {', '.join(columns)} FROM matrices").fetchall()
    if not rows:
        return []

    if folder is None:
        raise ValueError("Matrix metadata migration requires an on-disk project database")

    staged = []
    # Validate and edit copies before changing any original files.
    for file_name, *values in rows:
        path = folder / "matrices" / file_name
        metadata = {key: value for key, value in zip(columns, values, strict=True) if value is not None}
        if not path.is_file():
            if not metadata:
                logger.warning("Skipping missing matrix file %r: all associated metadata is NULL", str(path))
                continue
            raise FileNotFoundError(f"Cannot migrate matrix metadata: matrix file {str(path)!r} does not exist")

        copy = temporary / path.relative_to(folder / "matrices")
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, copy)
        with omx.open_file(copy, mode="r+") as matrix:
            conflicts = metadata.keys() & matrix.attrs.keys()
            if conflicts:
                raise ValueError(
                    f"Cannot migrate matrix metadata for {str(path)!r}: attributes already exist: {sorted(conflicts)}"
                )

            matrix.attrs.update(metadata)
        staged.append((copy, path))

    return staged


def move_results_table(project_conn: sqlite3.Connection, results_conn: Optional[sqlite3.Connection]):
    src_exists = _table_exists(project_conn, "results")
    num_rows = project_conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] if src_exists else 0
    if num_rows and results_conn is None:
        raise ValueError(f"There are {num_rows} rows in the results table, but results_conn is not given")

    # Check if it already exists.
    if results_conn is not None:
        dst_exists = _table_exists(results_conn, "results")
        if num_rows and dst_exists:
            # Possibly a previous run crashed after copying but before dropping.
            raise RuntimeError("results table exists in both the project and results databases")
        if dst_exists:
            add_triggers(results_conn, "results")
        else:
            initialize_tables("results", results_conn)

    if not num_rows:
        # It has no results yet. Drop the source table, and initialise the results database above if present.
        if src_exists:
            project_conn.execute("DROP TABLE results")
        return

    assert results_conn is not None and src_exists

    cols = [
        "table_name",
        "procedure",
        "procedure_id",
        "procedure_report",
        "timestamp",
        "description",
        "year",
        "scenario",
        "reference_table",
    ]
    col_names = ", ".join(cols)
    existing_columns = [row[1] for row in project_conn.execute('PRAGMA table_info("results")')]

    if set(cols) != set(existing_columns):
        raise ValueError(f"project database has different columns. Expected: {set(cols)}, got {set(existing_columns)}")
    data = project_conn.execute(f"SELECT {col_names} FROM results").fetchall()

    question_marks = ", ".join(["?" for _ in cols])
    results_conn.executemany(f"INSERT INTO results({col_names}) VALUES({question_marks})", data)

    project_conn.execute("DROP TABLE results")
