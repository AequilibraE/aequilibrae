"""Path-based project backups and imports which do not open or activate a Project."""

import logging
import shutil
import sqlite3
from contextlib import closing
from graphlib import TopologicalSorter
from os import PathLike
from pathlib import Path

logger = logging.getLogger(__name__)

_DATABASES = ("project_database.sqlite", "public_transport.sqlite", "results_database.sqlite")
_FILES = ("parameters.yml", "aequilibrae.log")
_FOLDERS = ("matrices", "run")
# Dependencies not expressed as foreign keys. Network link triggers inspect modes
# and nodes, and can create nodes which turn restrictions subsequently reference.
_COPY_DEPENDENCIES = {"links": {"nodes", "modes"}, "turn_restrictions": {"links", "modes"}}


def backup(source: str | PathLike, destination: str | PathLike) -> Path:
    """Back up a project and all its scenarios using SQLite's online backup API.

    Copy the three recognised databases before copying parameters.yml,
    aequilibrae.log, matrices/ and run/. Python caches and unrelated project-root
    files are excluded. The source is opened read-only, without opening a Project.

    The destination must not exist or be inside the source. On failure, stop and
    leave partial output in place. Each database is consistent independently;
    stop writers for a consistent snapshot across databases and other files.

    :Arguments:
        **source** (:obj:`str` or :obj:`os.PathLike`): Source project directory.
        **destination** (:obj:`str` or :obj:`os.PathLike`): New backup directory.

    :Returns:
        **destination** (:obj:`pathlib.Path`): Absolute backup directory.
    """
    return _copy_project(source, destination)


def copy_from_old(source: str | PathLike, destination: str | PathLike) -> Path:
    """Import an old project's network database into an existing destination schema.

    Both project_database.sqlite files must already exist. The actual destination
    schema determines which tables and writable columns are copied, including
    user-added schema. Missing source columns must have an explicit destination
    default, even if nullable; otherwise the import aborts. Source-only tables
    and columns are logged and omitted. Destination-only tables remain untouched.
    Generated columns are calculated by the destination, not copied.

    SQLite and SpatiaLite infrastructure and migration records are not copied.
    Source rows are inserted in dependency order using INSERT OR REPLACE, with
    foreign keys and insertion triggers enabled. Recursive triggers are disabled,
    so replacement does not run deletion-protection triggers. Initial rows without
    a source conflict remain. Use a freshly initialised destination: importing
    into populated projects is not supported or checked.

    The source is attached read-only; row data stays within SQLite. The import is
    one transaction, rolled back on failure. No databases, tables or accompanying
    files are created or copied. Transit, results and scenario databases are not
    imported, and references to their data are not checked. SpatiaLite is required
    when the destination is spatial. No migrations or renamed-field mappings run.

    :Arguments:
        **source** (:obj:`str` or :obj:`os.PathLike`): Old project directory.
        **destination** (:obj:`str` or :obj:`os.PathLike`): Existing new project directory.

    :Returns:
        **destination** (:obj:`pathlib.Path`): Absolute destination directory.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_database = source / "project_database.sqlite"
    destination_database = destination / "project_database.sqlite"
    for database in (source_database, destination_database):
        _reject_symlink(database)
        if not database.is_file():
            raise FileNotFoundError(f"Project database does not exist: {database}")
    if source_database.samefile(destination_database):
        raise ValueError("Source and destination databases must be different")
    try:
        _import_database(source_database, destination_database)
    except BaseException:
        logger.error("Import failed; no imported rows have been committed to %s", destination_database)
        raise
    logger.info("Completed project database import to %s", destination_database)
    return destination


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection, schema: str = "main") -> dict[str, str]:
    return dict(conn.execute(f"SELECT name, sql FROM {_quote(schema)}.sqlite_master WHERE type='table'"))


def _columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> dict[str, tuple[str | None, int]]:
    """Map column names to their default SQL and hidden/generated flag (zero means writable)."""
    return {
        row[1]: (row[4], row[6])
        for row in conn.execute(f"PRAGMA {_quote(schema)}.table_xinfo({_quote(table)})")
    }


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Symbolic links are not supported in project backups or imports: {path}")


def _scenario_folders(source: Path) -> list[Path]:
    folders = [source]
    scenarios = source / "scenarios"
    _reject_symlink(scenarios)
    if scenarios.is_dir():
        for folder in sorted(scenarios.iterdir()):
            if (folder / "project_database.sqlite").is_file():
                _reject_symlink(folder)
                folders.append(folder)
    return folders


def _copy_project(source: str | PathLike, destination: str | PathLike) -> Path:
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Destination must be outside the source project")
    if not (source / "project_database.sqlite").is_file():
        raise FileNotFoundError(f"Project database does not exist: {source / 'project_database.sqlite'}")
    folders = _scenario_folders(source)
    destination.mkdir(parents=True)
    try:
        # Complete all database copies before copying any accompanying files.
        for folder in folders:
            target = destination / folder.relative_to(source)
            target.mkdir(parents=True, exist_ok=True)
            for filename in _DATABASES:
                database = folder / filename
                _reject_symlink(database)
                if not database.is_file():
                    continue
                logger.info("Copying database %s to %s", database, target / filename)
                _backup_database(database, target / filename)
        for folder in folders:
            _copy_files(folder, destination / folder.relative_to(source))
    except BaseException:
        logger.error("Copy failed; partial output has been left at %s", destination)
        raise
    logger.info("Completed project copy to %s", destination)
    return destination


def _backup_database(source: Path, destination: Path) -> None:
    with closing(_read_only(source)) as src, closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)


def _copy_files(source: Path, destination: Path) -> None:
    for name in _FILES:
        path = source / name
        _reject_symlink(path)
        if path.is_file():
            shutil.copy2(path, destination / name)
    for name in _FOLDERS:
        path = source / name
        _reject_symlink(path)
        if path.is_dir():
            shutil.copytree(path, destination / name, ignore=_ignore_caches)


def _ignore_caches(directory: str, names: list[str]) -> list[str]:
    ignored = [name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))]
    for name in names:
        if name not in ignored:
            _reject_symlink(Path(directory) / name)
    return ignored


def _copy_statement(conn: sqlite3.Connection, table: str, database: Path) -> str:
    source_columns = _columns(conn, table, "old")
    target_columns = _columns(conn, table)
    writable = {column for column, (_, hidden) in target_columns.items() if hidden == 0}
    matching = writable & source_columns.keys()
    missing = writable - source_columns.keys()
    for column in sorted(source_columns.keys() - writable):
        reason = "generated or hidden in the destination" if column in target_columns else "absent from the destination"
        logger.info("Omitting column %s.%s from %s: %s", table, column, database, reason)
    for column, (_, hidden) in target_columns.items():
        if hidden != 0:
            logger.info("Not copying column %s.%s: generated or hidden in the destination", table, column)
    invalid = sorted(column for column in missing if target_columns[column][0] is None)
    for column in sorted(missing):
        default = target_columns[column][0]
        if default is None:
            logger.error(
                "Cannot copy table %s: source column %s is missing and has no destination default", table, column
            )
        else:
            logger.info("Using destination default for %s.%s: %s", table, column, default)
    if invalid:
        raise ValueError(f"Cannot copy table {table!r} from {database}: missing columns without defaults: {invalid}")
    columns = [column for column in target_columns if column in writable]
    if not columns:
        raise ValueError(f"Cannot copy table {table!r}: no writable destination columns")
    logger.info("Copying table %s from %s; matching columns: %s", table, database, sorted(matching))
    selection = ", ".join(
        f"source_row.{_quote(column)}" if column in matching else f"({target_columns[column][0]})"
        for column in columns
    )
    # Copy rows within SQLite rather than materialising potentially large network tables.
    return (
        f"INSERT OR REPLACE INTO main.{_quote(table)} ({', '.join(map(_quote, columns))}) "
        f"SELECT {selection} FROM old.{_quote(table)} AS source_row"
    )


def _check_database(conn: sqlite3.Connection, database: Path) -> None:
    violation = conn.execute("PRAGMA main.foreign_key_check").fetchone()
    if violation is not None:
        raise sqlite3.IntegrityError(f"Foreign-key violation in {database}: {violation}")
    checks = [row[0] for row in conn.execute("PRAGMA main.integrity_check")]
    if checks != ["ok"]:
        raise sqlite3.IntegrityError(f"Integrity check failed in {database}: {checks}")


def _infrastructure(conn: sqlite3.Connection) -> set[str]:
    # Infrastructure from the spatial template is never transferred. Include
    # spatial indexes of source-only layers and SQLite virtual-table shadow tables.
    from aequilibrae.reference_files import spatialite_database

    with closing(_read_only(Path(spatialite_database).resolve())) as template:
        infrastructure = set(_tables(template)) | {"migrations"}
    for schema in ("main", "old"):
        tables = _tables(conn, schema)
        infrastructure.update(table for table in tables if table.startswith("sqlite_"))
        infrastructure.update(
            row[1] for row in conn.execute(f"PRAGMA {_quote(schema)}.table_list") if row[2] == "shadow"
        )
        if "geometry_columns" in tables:
            for table, column in conn.execute(
                f"SELECT f_table_name, f_geometry_column FROM {_quote(schema)}.geometry_columns"
            ):
                index = f"idx_{table}_{column}"
                infrastructure.update(index + suffix for suffix in ("", "_node", "_parent", "_rowid"))
    return infrastructure


def _table_copy_order(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """Order parents before children, including dependencies used by editing triggers.

    Sort all destination tables, including those retaining their initial rows. A
    cycle is an error rather than an excuse to disable constraints or guess an order.
    """
    available = set(tables)
    dependencies = {}
    for table in tables:
        parents = {row[2] for row in conn.execute(f"PRAGMA main.foreign_key_list({_quote(table)})")}
        dependencies[table] = sorted((parents | _COPY_DEPENDENCIES.get(table, set())) & available)
    return list(TopologicalSorter(dependencies).static_order())


def _import_database(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(destination.as_uri() + "?mode=rw", uri=True)) as conn:
        if "geometry_columns" in _tables(conn):
            # Lazy imports keep plain SQLite backups independent of spatial initialisation.
            from aequilibrae.utils.spatialite_utils import ensure_spatialite_binaries, load_spatialite_extension

            ensure_spatialite_binaries()
            load_spatialite_extension(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA recursive_triggers = OFF")
        conn.execute("ATTACH DATABASE ? AS old", (source.as_uri() + "?mode=ro",))
        with conn:
            conn.execute("BEGIN")
            source_tables = _tables(conn, "old")
            target_tables = _tables(conn)
            infrastructure = _infrastructure(conn)
            for table in sorted(source_tables.keys() | target_tables.keys()):
                if table in infrastructure:
                    logger.info("Not copying table %s: SQLite/SpatiaLite infrastructure or migration records", table)
                elif table not in target_tables:
                    logger.info("Omitting table %s from %s: absent from the destination", table, source)
                elif table not in source_tables:
                    logger.info("Leaving destination table %s untouched: absent from %s", table, source)
            ordered = _table_copy_order(conn, sorted(target_tables.keys() - infrastructure))
            statements = {table: _copy_statement(conn, table, source) for table in ordered if table in source_tables}
            for table, statement in statements.items():
                try:
                    conn.execute(statement)
                except sqlite3.Error as error:
                    error.add_note(f"While copying table {table!r} from {source}")
                    raise
            _check_database(conn, destination)
