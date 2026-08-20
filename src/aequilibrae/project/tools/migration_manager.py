import logging
import pathlib
import sqlite3
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from aequilibrae.utils.db_utils import ConnectionClosure
from aequilibrae.utils.model_run_utils import import_file_as_module

logger = logging.getLogger(__name__)


class MigrationStatus(IntEnum):
    MISSING = 1
    SKIPPED = 2
    APPLIED = 3


@dataclass
class Migration:
    """
    Small utility class to wrap files used for database upgrades/migrations.

    Individual migrations can report their status, be marked as 'seen' or as
    another status, and applied. Migrations are Python modules that expose a
    ``migrate`` function accepting ``project_conn``, ``transit_conn``, and
    ``results_conn`` keyword arguments for the open database connections.

    Marking a migration as 'seen' will add it to the ``migrations`` table as
    ``MISSING`` if it is not already present. If it is present no change is made.

    Applying a migration will update the status to 'APPLIED' with the current
    timestamp.

    A migration's status cannot be downgraded without force.

    Migrations are identified based on their ``id`` attribute and the ``id``
    field of the ``migrations`` table.
    """

    id: int
    name: str
    file: pathlib.Path

    def __post_init__(self):
        if self.file.suffix != ".py":
            raise ValueError("only '.py' files are supported for migrations")

    def status(self, conn: sqlite3.Connection) -> MigrationStatus:
        """
        Query the database for this migration's status.

        If the ``migrations`` table is not present all migrations are considered ``MISSING``.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.

        :Returns:
            **status** (:obj:`MigrationStatus`): Migration status enum.
        """
        res = conn.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
        return MigrationStatus.MISSING if res is None else MigrationStatus[res[0]]

    def mark_as(self, conn: sqlite3.Connection, status: MigrationStatus, force: bool = False):
        """
        Update or insert this migration with the given status.

        If the migration is not present in the table it will be inserted. If it is present and the new status is an
        'upgrade' or ``force=True``, then it will be updated. Otherwise no change will be made.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.

            **status** (:obj:`MigrationStatus`): Migration status enum.
        """
        res = conn.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
        if res is None:
            conn.execute(
                "INSERT INTO migrations (id, name, status, date) VALUES(?,?,?,CURRENT_TIMESTAMP)",
                (self.id, self.name, status.name),
            )
            return
        previous = MigrationStatus[res[0]]
        # Allow marking the status as APPLIED when it is MISSING or SKIPPED, and
        # as SKIPPED when it is MISSING, or any change whenever force is True.
        if force or previous < status or previous < status < MigrationStatus.APPLIED:
            conn.execute(
                "UPDATE migrations SET status=?, name=?, date=CURRENT_TIMESTAMP WHERE id=?",
                (status.name, self.name, self.id),
            )

    def mark_as_seen(self, conn: sqlite3.Connection):
        """
        Mark this migration as 'seen'.

        Marking a migration as 'seen' will add it to the ``migrations`` table as ``MISSING`` if it is not already
        present. If it is present no change is made.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.
        """
        self.mark_as(conn, MigrationStatus.MISSING, force=False)

    def apply(self, conn: sqlite3.Connection, connections: dict[str, Optional[sqlite3.Connection]]):
        """
        Apply this migration.

        Successful application will mark the migration as ``APPLIED``.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): Main SQLite database connection. Used for the migrations table.

            **connections** (:obj:`dict[str, Optional[sqlite3.Connection]]`): Named SQLite connections. Passed as
            keyword arguments to the migration's ``migrate`` function.
        """
        self._apply_python(connections)
        self.mark_as(conn, MigrationStatus.APPLIED)
        logger.info("Completed migration '%s'", self.name)

    def _apply_python(self, connections: dict[str, Optional[sqlite3.Connection]]):
        module = import_file_as_module(self.file, self.name, force=True)
        try:
            migrate = module.migrate
        except AttributeError as error:
            raise RuntimeError(f"'{self.name} does not expose a global 'migrate' callable") from error

        if not callable(migrate):
            raise RuntimeError("found 'migrate' symbol in the migration file but it is not callable")

        migrate(**connections)


class MigrationManager:
    r"""
    Small utility class to manage, validate, and apply a set of ``Migration``\s.

    :Arguments:
        **migration_file** (:obj:`pathlib.Path`): A path to a Python file which defines a global ``migrations``
        variable as a list of ``pathlib.Path`` objects.

        **database** (:obj:`str`, optional): Name of the closure connection that owns the ``migrations`` table.
        Defaults to ``'transit'`` when the migration file lives under a transit directory and ``'project'`` otherwise.
    """

    network_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "network" / "migrations" / "migrations.py"
    )
    transit_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "transit" / "migrations" / "migrations.py"
    )

    def __init__(self, migration_file: pathlib.Path, database: str | None = None):
        migration_file = pathlib.Path(migration_file)

        if database is None:
            parent_names = {part.name for part in migration_file.parents}
            database = "transit" if "transit" in parent_names else "project"
        self.database = database

        files = import_file_as_module(
            migration_file, "aequilibrae.project.database_specification.migrations", force=True
        ).migrations

        migrations = []
        for file in files:
            if not file.exists():
                raise FileNotFoundError(f"migration file '{file.name}' does not exist")

            identifier, _, name = file.stem.partition("_")
            identifier = int(identifier)

            if identifier < 0:
                raise ValueError("migration IDs must be >= 0")
            migrations.append(Migration(identifier, name, file))

        self.migrations = {migration.id: migration for migration in sorted(migrations, key=lambda item: item.id)}

        if len(self.migrations) != len(migrations):
            raise ValueError("duplicate migration IDs found. Ensure migration IDs are unique.")

    def _connections(self, closure: ConnectionClosure) -> dict[str, Optional[sqlite3.Connection]]:
        """Return the raw connections expected by Python migration functions."""
        return {
            "project_conn": closure.db_connection.connection,
            "transit_conn": closure.transit_connection.connection if closure.has_transit_connection else None,
            "results_conn": closure.results_connection.connection if closure.has_results_connection else None,
        }

    def _database_connection(self, closure: ConnectionClosure) -> sqlite3.Connection:
        """Return the connection which owns this migration series."""
        if self.database == "project":
            return closure.db_connection.connection
        if self.database == "transit":
            return closure.transit_connection.connection
        raise ValueError(f"unknown migration database: {self.database}")

    def _ensure_initial_is_applied(
        self, conn: sqlite3.Connection, connections: dict[str, Optional[sqlite3.Connection]]
    ):
        # Handle the initial migration separately: the 'migrations' table might not have been created yet. We
        # implicitly apply this migration all the time to ensure the table exists.
        table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations'").fetchone()
        if table is None:
            if 0 not in self.migrations:
                raise RuntimeError("migration series has no initial migration")
            self.migrations[0].apply(conn, connections)

    def status(self, closure: ConnectionClosure) -> dict[int, MigrationStatus]:
        """
        Query the database for all migrations' status.

        If the ``migrations`` table is not present all migrations are considered ``MISSING``.

        :Arguments:
            **closure** (:obj:`ConnectionClosure`): The scenario's connection closure.

        :Returns:
            **status** (:obj:`dict[int, MigrationStatus]`): Migration status enums by their ID.
        """
        with closure.transaction():
            conn = self._database_connection(closure)
            connections = self._connections(closure)
            self._ensure_initial_is_applied(conn, connections)
            return {key: migration.status(conn) for key, migration in self.migrations.items()}

    def mark_all_as_seen(self, closure: ConnectionClosure):
        """
        Mark all migrations as 'seen'.

        Marking a migration as 'seen' will add it to the ``migrations`` table as ``MISSING`` if it is not already
        present. If it is present no change is made.

        :Arguments:
            **closure** (:obj:`ConnectionClosure`): The scenario's connection closure.
        """
        with closure.transaction():
            conn = self._database_connection(closure)
            connections = self._connections(closure)
            self._ensure_initial_is_applied(conn, connections)
            for migration in self.migrations.values():
                migration.mark_as_seen(conn)

    def _find_applicable(self, conn: sqlite3.Connection):
        statuses = [(key, migration.status(conn)) for key, migration in self.migrations.items()]
        first_missing = len(statuses)
        for index, (_, status) in enumerate(statuses):
            if status == MigrationStatus.MISSING:
                first_missing = index
                break
        applicable = []
        for key, status in statuses[first_missing:]:
            if status == MigrationStatus.APPLIED:
                raise RuntimeError("out of order migration application found. Manual intervention required")
            applicable.append(self.migrations[key])
        return applicable

    def find_applicable(self, closure: ConnectionClosure):
        """
        Find all applicable migrations.

        A migration is applicable if all migrations before it (ordered by ID) have been applied or skipped.

        If an out-of-order migration is detected a ``RuntimeError`` will be raised and manual intervention will be
        required.

        :Arguments:
            **closure** (:obj:`ConnectionClosure`): The scenario's connection closure.
        """
        with closure.transaction():
            conn = self._database_connection(closure)
            connections = self._connections(closure)
            self._ensure_initial_is_applied(conn, connections)
            return self._find_applicable(conn)

    def upgrade(self, closure: ConnectionClosure, skip: set[int] | None = None):
        """
        Find and apply all applicable migrations in one closure-owned transaction.

        Optionally skip some migrations. Take care when skipping migrations.

        :Arguments:
            **closure** (:obj:`ConnectionClosure`): The scenario's connection closure.

            **skip** (:obj:`set[int]`): Set of migration IDs to skip.
        """
        if skip is None:
            skip = set()

        with closure.transaction():
            conn = self._database_connection(closure)
            connections = self._connections(closure)
            self._ensure_initial_is_applied(conn, connections)

            for migration in self._find_applicable(conn):
                if migration.id in skip:
                    migration.mark_as(conn, MigrationStatus.SKIPPED)
                else:
                    migration.apply(conn, connections)
