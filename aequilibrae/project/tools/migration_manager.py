import pathlib
import sqlite3
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from aequilibrae import logger
from aequilibrae.utils.model_run_utils import import_file_as_module
from aequilibrae.utils.db_utils import AequilibraEConnection


class MigrationStatus(IntEnum):
    MISSING: int = 1
    SKIPPED: int = 2
    APPLIED: int = 3


@dataclass
class Migration:
    """
    Small utility class to wrap files used for database upgrades/migrations.

    Individual migrations can report their status, be marked as 'seen' or as another status, and applied. SQL migrations
    are executed using ``sqlite3.executescript``. Python migrations are loaded as a module, they should expose a
    ``migrate`` function which accepts an ``sqlite3.Connection`` as a single positional argument.

    Marking a migration as 'seen' will add it to the ``migrations`` table as ``MISSING`` if it is not already
    present. If it is present no change is made.

    Applying a migration will update the status to 'APPLIED' with the current timestamp.

    A migration's status cannot be downgraded without force.

    Migrations are identified based on their ``id`` attribute and the ``id`` field of the ``migrations`` table.
    """

    id: int
    name: str
    file: pathlib.Path
    type: str = None

    def __post_init__(self):
        if self.file.suffix == ".py":
            self.type = "py"
        elif self.file.suffix == ".sql":
            self.type = "sql"
        else:
            raise ValueError("only Python ('.py') and SQL ('.sql') files are supported for migrations")

    def status(self, conn: sqlite3.Connection) -> MigrationStatus:
        """
        Query the database for this migrations status.

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

        If the migration is not present in the table it will be inserted. If it is present and the new status is a
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
        else:
            res = MigrationStatus[res[0]]
            if force or res < status or res < status < MigrationStatus.APPLIED:
                # We want to allow marking the status as APPLIED if it is MISSING or SKIPPED, and as SKIPPED if it
                # is MISSING, or just whenever force is True
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

    def apply(self, conn: sqlite3.Connection):
        """
        Apply this migration.

        Successful application will mark the migration as ``APPLIED``.

        Python migrations should never use ``executescript`` as it will commit the pending transaction and place SQLite
        in autocommit mode. If the migration then fails the database will be bad state.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.
        """
        if self.type == "py":
            self._apply_python(conn)
        elif self.type == "sql":
            self._apply_sql(conn)
        else:
            raise ValueError("only Python ('.py') and SQL ('.sql') files are supported for migrations")

        self.mark_as(conn, MigrationStatus.APPLIED)
        logger.info(f"Completed migration '{self.name}'")

    def _apply_sql(self, conn: sqlite3.Connection):
        with open(self.file, "r") as f:
            contents = f.read()
        conn.executescript(contents)

    def _apply_python(self, conn: sqlite3.Connection):
        module = import_file_as_module(self.file, self.name, force=True)
        try:
            migrate = module.migrate
        except AttributeError as e:
            raise RuntimeError(f"'{self.name} does not expose a global 'migrate' callable") from e

        if not callable(migrate):
            raise RuntimeError("found 'migrate' symbol in the migration file but it is not callable")

        migrate(conn)


class MigrationManager:
    """
    Small utility class to manage, validate, and apply a set of ``Migration``s.

    :Arguments:
        **migration_file** (:obj:`pathlib.Path`): A path to a Python with which defines a global ``migrations`` variable
            as a list of ``pathlib.Path`` to migrations.
    """

    network_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "network" / "migrations" / "migrations.py"
    )
    transit_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "transit" / "migrations" / "migrations.py"
    )

    def __init__(self, migration_file: pathlib.Path):
        migrations = import_file_as_module(
            migration_file,
            "aequilibrae.project.database_specification.migrations",
            force=True,
        ).migrations

        res = []
        for migration in migrations:
            if not migration.exists():
                raise FileNotFoundError(f"migration file '{migration.name}' does not exist'")

            id, _, name = migration.stem.partition("_")
            id = int(id)
            if id < 0:
                raise ValueError("migration IDs must be >= 0")
            res.append(Migration(id=id, name=name, file=migration))

        self.migrations: dict[int, Migration] = {
            migration.id: migration for migration in sorted(res, key=lambda x: x.id)
        }
        if len(self.migrations) != len(res):
            raise ValueError("duplicate migration IDs found. Ensure migration IDs are unique.")

    def __ensure_inital_is_applied(self, conn):
        # Handle the initial migration separately, the 'migrations' table might not have been created. We implicitly
        # apply this migration all the time to ensure the table exists.
        with conn:
            self.migrations[0].apply(conn)

    def status(self, conn: sqlite3.Connection) -> dict[int, MigrationStatus]:
        """
        Query the database for all migrations' status.

        If the ``migrations`` table is not present all migrations are considered ``MISSING``.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.

        :Returns:
            **status** (:obj:`dict[int, MigrationStatus]`): Migration status enums by their ID.
        """
        self.__ensure_inital_is_applied(conn)
        return {k: v.status(conn) for k, v in self.migrations.items()}

    def mark_all_as_seen(self, conn: sqlite3.Connection):
        """
        Mark all migrations as 'seen'.

        Marking a migration as 'seen' will add it to the ``migrations`` table as ``MISSING`` if it is not already
        present. If it is present no change is made.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.
        """
        self.__ensure_inital_is_applied(conn)
        with conn:
            for migration in self.migrations.values():
                migration.mark_as_seen(conn)

    def find_applicable(self, conn: sqlite3.Connection):
        """
        Find all applicable migrations.

        A migration is applicable if all migrations before it (ordered by ID) have been applied or skipped.

        If an out-of-order migration is detected a ``RuntimeError` will raised and manual intervention will be required.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.
        """
        migrations = list(self.status(conn).items())

        for i in range(len(migrations)):
            k, v = migrations[i]
            if v == MigrationStatus.MISSING:
                break
        else:
            i += 1

        res = []
        for j in range(i, len(migrations)):
            k, v = migrations[j]
            if v == MigrationStatus.APPLIED:
                raise RuntimeError("out of order migration application found. Manual intervention required")
            else:
                res.append(self.migrations[k])

        return res

    def upgrade(self, conn: AequilibraEConnection, skip: set[int] = None):
        """
        Find and apply all applicable migrations.

        Optionally skip some migrations. Take care when skipping migrations.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): SQLite database connection.
            **skip** (:obj:`set[int]`): Set of migration IDs to skip.
        """
        if skip is None:
            skip = set()
        migrations = self.find_applicable(conn)

        for migration in migrations:
            with conn.manual_transaction():
                if migration.id in skip:
                    migration.mark_as(conn, MigrationStatus.SKIPPED)
                else:
                    migration.apply(conn)
