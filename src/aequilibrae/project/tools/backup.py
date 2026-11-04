"""Path-based project backups which do not open or activate a Project."""

import logging
import shutil
import sqlite3
from contextlib import closing
from graphlib import TopologicalSorter
from os import PathLike
from pathlib import Path

logger = logging.getLogger(__name__)

_DATABASES = {
    "project_database.sqlite": "network",
    "public_transport.sqlite": "transit",
    "results_database.sqlite": None,
}
_FILES = ("parameters.yml", "aequilibrae.log")
_FOLDERS = ("matrices", "run")
_SPECIFICATION = Path(__file__).resolve().parent.parent / "database_specification"
# Dependencies not expressed as foreign keys. Network link triggers inspect modes
# and nodes, and can create nodes which turn restrictions subsequently reference.
# Transit graph links also depend on graph nodes and their lookup/settings tables.
_COPY_DEPENDENCIES = {
    "network": {"links": {"nodes", "modes"}, "turn_restrictions": {"links", "modes"}},
    "transit": {"nodes": {"node_types"}, "links": {"nodes", "modes", "trigger_settings"}},
}


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
    return _copy_project(source, destination, rebuild=False)


def copy_to_new(source: str | PathLike, destination: str | PathLike) -> Path:
    """Copy a project and all its scenarios into the installed version's schema.

    Copy matching managed tables and columns, logging omitted source tables and
    columns. No migrations or renamed-field mappings are applied. New columns use
    their schema defaults. New tables retain their initial data; existing tables
    receive source data instead of their initial rows. Migration records and
    SpatiaLite infrastructure are freshly initialised, not copied. Triggers are
    installed before inserting source rows in dependency order. Trigger-generated
    changes, including recalculated fields and additional nodes, are accepted.

    Registered result tables have no fixed schema, so their definitions, data
    and indexes are recreated. Unregistered result tables are logged and omitted.
    Constraints and registered scenario/result dependencies are checked; failures
    stop the copy, leaving partial output.
    This is not a substitute for an exact backup before upgrading a project.

    Paths, accompanying files and snapshot limitations are as for :func:`backup`.
    SpatiaLite is required for this operation, but not for plain SQLite backups.

    :Arguments:
        **source** (:obj:`str` or :obj:`os.PathLike`): Source project directory.
        **destination** (:obj:`str` or :obj:`os.PathLike`): New project directory.

    :Returns:
        **destination** (:obj:`pathlib.Path`): Absolute new project directory.
    """
    return _copy_project(source, destination, rebuild=True)


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"))


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    """Map column names to SQLite's hidden/generated flag (zero means writable)."""
    return {row[1]: row[6] for row in conn.execute(f"PRAGMA table_xinfo({_quote(table)})")}


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Symbolic links are not supported in project backups: {path}")


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


def _copy_project(source: str | PathLike, destination: str | PathLike, *, rebuild: bool) -> Path:
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
            for filename, schema in _DATABASES.items():
                database = folder / filename
                _reject_symlink(database)
                if not database.is_file():
                    continue
                logger.info("Copying database %s to %s", database, target / filename)
                if not rebuild:
                    _backup_database(database, target / filename)
                elif schema is not None:
                    _copy_schema_database(database, target / filename, schema, scenario=folder != source)
                else:
                    _copy_results(database, target / filename, target / "project_database.sqlite")
        if rebuild:
            _check_project_references([destination / folder.relative_to(source) for folder in folders])
        for folder in folders:
            _copy_files(folder, destination / folder.relative_to(source))
    except BaseException:
        logger.error("Copy failed; partial output has been left at %s", destination)
        raise
    logger.info("Completed project copy to %s", destination)
    return destination


def _check_project_references(folders: list[Path]) -> None:
    """Reject dangling scenario/result records in rebuilt projects, not plain backups."""
    available_scenarios = {"root"} | {folder.name for folder in folders[1:]}
    for folder in folders:
        with closing(_read_only(folder / "project_database.sqlite")) as conn:
            if folder == folders[0]:
                registered = {row[0] for row in conn.execute("SELECT scenario_name FROM scenarios")}
                missing = registered - available_scenarios
                if missing:
                    raise FileNotFoundError(f"Registered scenario databases missing from {folder}: {sorted(missing)}")
            if not (folder / "results_database.sqlite").is_file():
                registered = {row[0] for row in conn.execute("SELECT table_name FROM results")}
                if registered:
                    raise FileNotFoundError(
                        f"Results database missing from {folder}; registered tables: {sorted(registered)}"
                    )


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


def _copy_rows(src: sqlite3.Connection, dst: sqlite3.Connection, table: str, database: Path) -> None:
    source_columns = _columns(src, table)
    target_columns = _columns(dst, table)
    for column in source_columns:
        if column not in target_columns:
            logger.warning("Omitting column %s.%s from %s: absent from the new schema", table, column, database)
    columns = [column for column in source_columns if target_columns.get(column) == 0]
    quoted_table = _quote(table)
    try:
        if columns:
            selection = ", ".join(map(_quote, columns))
            placeholders = ", ".join("?" for _ in columns)
            # Stream rows rather than materialising potentially large network/result tables.
            dst.executemany(
                f"INSERT INTO {quoted_table} ({selection}) VALUES ({placeholders})",
                src.execute(f"SELECT {selection} FROM {quoted_table}"),
            )
        else:
            for _ in src.execute(f"SELECT 1 FROM {quoted_table}"):
                dst.execute(f"INSERT INTO {quoted_table} DEFAULT VALUES")
    except sqlite3.Error as error:
        error.add_note(f"While copying table {table!r} from {database}")
        raise


def _check_database(conn: sqlite3.Connection, database: Path) -> None:
    violation = conn.execute("PRAGMA foreign_key_check").fetchone()
    if violation is not None:
        raise sqlite3.IntegrityError(f"Foreign-key violation in {database}: {violation}")
    checks = [row[0] for row in conn.execute("PRAGMA integrity_check")]
    if checks != ["ok"]:
        raise sqlite3.IntegrityError(f"Integrity check failed in {database}: {checks}")


def _spatial_tables(src: sqlite3.Connection, dst: sqlite3.Connection, managed: list[str]) -> set[str]:
    # Infrastructure already present in a freshly initialised spatial database is
    # rebuilt, never transferred. Include spatial indexes of source-only layers.
    infrastructure = set(_tables(dst)) - set(managed)
    if "geometry_columns" in _tables(src):
        for table, column in src.execute("SELECT f_table_name, f_geometry_column FROM geometry_columns"):
            index = f"idx_{table}_{column}"
            infrastructure.update(index + suffix for suffix in ("", "_node", "_parent", "_rowid"))
    return infrastructure


def _table_copy_order(conn: sqlite3.Connection, tables: list[str], schema: str) -> list[str]:
    """Order parents before children, including dependencies used by editing triggers.

    Sort all managed tables, including those retaining their initial rows. A cycle
    is an error rather than an excuse to disable constraints or guess an order.
    """
    managed = set(tables)
    extra = _COPY_DEPENDENCIES.get(schema, {})
    dependencies = {}
    for table in tables:
        parents = {row[2] for row in conn.execute(f"PRAGMA foreign_key_list({_quote(table)})")}
        dependencies[table] = sorted((parents | extra.get(table, set())) & managed)
    return list(TopologicalSorter(dependencies).static_order())


def _copy_schema_database(source: Path, destination: Path, schema: str, *, scenario: bool) -> None:
    # Lazy imports avoid a project_creation -> tools -> backup import cycle and
    # keep the plain backup implementation independent of spatial initialisation.
    from aequilibrae.project.project_creation import add_triggers, create_base_tables
    from aequilibrae.reference_files import spatialite_database
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    shutil.copyfile(spatialite_database, destination)
    with closing(_read_only(source)) as src, closing(connect_spatialite(destination)) as dst:
        src.execute("BEGIN")
        dst.execute("PRAGMA foreign_keys = ON")
        create_base_tables(dst, schema)
        if schema == "network" and scenario:
            dst.execute("DROP TABLE scenarios")
        dst.commit()
        managed = (_SPECIFICATION / schema / "tables" / "table_list.txt").read_text().splitlines()
        target_tables = _tables(dst)
        managed = [table for table in managed if table in target_tables]
        source_tables = _tables(src)
        infrastructure = _spatial_tables(src, dst, managed)
        for table in sorted(source_tables):
            if table not in managed and table not in infrastructure and not table.startswith("sqlite_"):
                logger.warning("Omitting table %s from %s: absent from the new schema", table, source)
        ordered = _table_copy_order(dst, [table for table in managed if table != "migrations"], schema)
        tables = [table for table in ordered if table in source_tables]
        # Clear initial rows before installing editing triggers, which protect
        # default modes, link types, periods and the root scenario from deletion.
        # Foreign keys, geometry constraints and spatial-index triggers stay
        # enabled. Insert parents first, accepting editing-trigger changes.
        with dst:
            dst.execute("BEGIN")
            for table in reversed(tables):
                dst.execute(f"DELETE FROM {_quote(table)}")
            add_triggers(dst, schema, skip={"scenarios_triggers"} if scenario and schema == "network" else None)
            for table in tables:
                _copy_rows(src, dst, table, source)
            _check_database(dst, destination)


def _copy_results(source: Path, destination: Path, project_database: Path) -> None:
    with closing(_read_only(project_database)) as project:
        registered = {row[0] for row in project.execute("SELECT table_name FROM results")}
    with closing(_read_only(source)) as src, closing(sqlite3.connect(destination)) as dst:
        src.execute("BEGIN")
        tables = _tables(src)
        for table in sorted(tables):
            if table not in registered and not table.startswith("sqlite_"):
                logger.warning("Omitting unregistered result table %s from %s", table, source)
        missing = registered - tables.keys()
        if missing:
            raise ValueError(f"Registered result tables missing from {source}: {sorted(missing)}")
        with dst:
            dst.execute("BEGIN")
            for table in sorted(registered):
                definition = tables[table]
                if definition is None or "CREATE VIRTUAL TABLE" in definition.upper():
                    raise ValueError(f"Unsupported result table {table!r} in {source}")
                dst.execute(definition)
                _copy_rows(src, dst, table, source)
                for (sql,) in src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)
                ):
                    dst.execute(sql)
            _check_database(dst, destination)
