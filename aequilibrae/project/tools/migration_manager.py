import pathlib
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from aequilibrae import logger
from aequilibrae.utils.model_run_utils import import_file_as_module


class MigrationStatus(Enum):
    APPLIED: str = "APPLIED"
    MISSING: str = "MISSING"
    SKIPPED: str = "SKIPPED"


@dataclass
class Migration:
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
        res = conn.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
        return MigrationStatus.MISSING if res is None else MigrationStatus(res[0])

    def mark_as(self, conn: sqlite3.Connection, status: MigrationStatus, force: bool = False):
        with conn as conn:
            res = conn.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
            if res is None:
                conn.execute(
                    "INSERT INTO migrations (id, name, status, date) VALUES(?,?,?,CURRENT_TIMESTAMP)",
                    (self.id, self.name, status.name),
                )
            elif force:
                conn.execute(
                    "UPDATE migrations SET status=?, name=?, date=CURRENT_TIMESTAMP WHERE id=?",
                    (status.name, self.name, self.id),
                )

    def mark_as_seen(self, conn: sqlite3.Connection):
        self.mark_as(conn, MigrationStatus.MISSING, force=False)

    def apply(self, conn: sqlite3.Connection):
        logger.info(f"Applying migration '{self.name}'")
        with conn as conn:
            if self.type == "py":
                self._apply_python(conn)
            elif self.type == "sql":
                self._apply_sql(conn)
            else:
                raise ValueError("only Python ('.py') and SQL ('.sql') files are supported for migrations")
        logger.info(f"Completed migration '{self.name}'")

    def _apply_sql(self, conn: sqlite3.Connection):
        with open(self.file, "r") as f:
            contents = f.read()
        conn.executescript(contents)

    def _apply_python(self, conn: sqlite3.Connection):
        module = import_file_as_module(self.file, self.name)
        try:
            migrate = module.migrate
        except AttributeError as e:
            raise RuntimeError(f"'{self.name} does not expose a global 'migrate' callable") from e

        if not callable(migrate):
            raise RuntimeError("found 'migrate' symbol in the migration file but it is not callable")

        migrate(conn)


class MigrationManager:
    network_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "network" / "migrations" / "__init__.py"
    )
    transit_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "transit" / "migrations" / "__init__.py"
    )

    def __init__(self, migration_file: pathlib.Path):
        migrations = import_file_as_module(
            migration_file,
            "aequilibrae.project.database_specification.migrations",
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
        with conn as _conn:
            self.migrations[0].apply(_conn)
            self.migrations[0].mark_as(_conn, MigrationStatus.APPLIED)

    def status(self, conn: sqlite3.Connection) -> dict[int, MigrationStatus]:
        self.__ensure_inital_is_applied(conn)
        return {k: v.status(conn) for k, v in self.migrations.items()}

    def mark_all_as_seen(self, conn: sqlite3.Connection):
        self.__ensure_inital_is_applied(conn)
        with conn as conn:
            for migration in self.migrations.values():
                migration.mark_as_seen(conn)

    def find_applicable(self, conn: sqlite3.Connection):
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

    def upgrade(self, conn: sqlite3.Connection, skip: set[int] = None):
        if skip is None:
            skip = set()
        migrations = self.find_applicable(conn)

        with conn as conn:
            for migration in migrations:
                if migration.id in skip:
                    migration.mark_as(conn, MigrationStatus.SKIPPED)
                else:
                    migration.apply(conn)
                    migration.mark_as(conn, MigrationStatus.APPLIED)
