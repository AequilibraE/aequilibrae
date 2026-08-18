import logging
import pathlib
import re
import sqlite3
from dataclasses import dataclass
from enum import IntEnum

from aequilibrae.utils.db_utils import ConnectionClosure, NestedTransactions
from aequilibrae.utils.model_run_utils import import_file_as_module

logger = logging.getLogger(__name__)


class MigrationStatus(IntEnum):
    MISSING = 1
    SKIPPED = 2
    APPLIED = 3


@dataclass
class Migration:
    """One SQL or Python schema migration."""

    id: int
    name: str
    file: pathlib.Path
    database: str
    type: str = None

    def __post_init__(self):
        if self.file.suffix not in (".py", ".sql"):
            raise ValueError("only '.py' and '.sql' files are supported for migrations")
        self.type = self.file.suffix[1:]

    def status(self, manager: NestedTransactions) -> MigrationStatus:
        row = manager.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
        return MigrationStatus.MISSING if row is None else MigrationStatus[row[0]]

    def mark_as(self, manager: NestedTransactions, status: MigrationStatus, force: bool = False):
        row = manager.execute("SELECT status FROM migrations WHERE id=?", (self.id,)).fetchone()
        if row is None:
            manager.execute(
                "INSERT INTO migrations (id, name, status, date) VALUES(?,?,?,CURRENT_TIMESTAMP)",
                (self.id, self.name, status.name),
            )
            return
        previous = MigrationStatus[row[0]]
        if force or previous < status or previous < status < MigrationStatus.APPLIED:
            manager.execute(
                "UPDATE migrations SET status=?, name=?, date=CURRENT_TIMESTAMP WHERE id=?",
                (status.name, self.name, self.id),
            )

    def mark_as_seen(self, manager: NestedTransactions):
        self.mark_as(manager, MigrationStatus.MISSING)

    def apply(self, closure: ConnectionClosure):
        manager = closure[self.database]
        if self.type == "sql":
            self._apply_sql(manager)
        else:
            self._apply_python(closure)
        self.mark_as(manager, MigrationStatus.APPLIED)
        logger.info("Completed migration '%s'", self.name)

    def _apply_sql(self, manager: NestedTransactions):
        for statement in iter_sql_statements(self.file.read_text()):
            manager.execute(statement)

    def _apply_python(self, closure: ConnectionClosure):
        module = import_file_as_module(self.file, self.name, force=True)
        try:
            migrate = module.migrate
        except AttributeError as error:
            raise RuntimeError(f"'{self.name} does not expose a global 'migrate' callable") from error
        if not callable(migrate):
            raise RuntimeError("found 'migrate' symbol in the migration file but it is not callable")
        migrate(closure=closure)


class MigrationManager:
    """Apply a migration series through an owning named connection closure."""

    network_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "network" / "migrations" / "migrations.py"
    )
    transit_migration_file = (
        pathlib.Path(__file__).parent.parent / "database_specification" / "transit" / "migrations" / "migrations.py"
    )

    def __init__(self, migration_file: pathlib.Path, database: str = None):
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
            migrations.append(Migration(identifier, name, file, database))
        self.migrations = {migration.id: migration for migration in sorted(migrations, key=lambda item: item.id)}
        if len(self.migrations) != len(migrations):
            raise ValueError("duplicate migration IDs found. Ensure migration IDs are unique.")

    def _initialize_status(self, closure: ConnectionClosure):
        manager = closure[self.database]
        table = manager.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations'"
        ).fetchone()
        if table is None:
            if 0 not in self.migrations:
                raise RuntimeError("migration series has no initial migration")
            self.migrations[0].apply(closure)

    def status(self, closure: ConnectionClosure) -> dict[int, MigrationStatus]:
        with closure.transaction():
            self._initialize_status(closure)
            manager = closure[self.database]
            return {key: migration.status(manager) for key, migration in self.migrations.items()}

    def mark_all_as_seen(self, closure: ConnectionClosure):
        with closure.transaction():
            self._initialize_status(closure)
            manager = closure[self.database]
            for migration in self.migrations.values():
                migration.mark_as_seen(manager)

    def _find_applicable(self, manager: NestedTransactions):
        statuses = [(key, migration.status(manager)) for key, migration in self.migrations.items()]
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
        with closure.transaction():
            self._initialize_status(closure)
            return self._find_applicable(closure[self.database])

    def upgrade(self, closure: ConnectionClosure):
        """Apply schema and status writes in one closure-owned transaction."""
        with closure.transaction():
            self._initialize_status(closure)
            for migration in self._find_applicable(closure[self.database]):
                migration.apply(closure)


def iter_sql_statements(sql: str):
    """Yield complete SQLite statements without using ``executescript``.

    ``sqlite3.complete_statement`` understands trigger bodies containing
    internal semicolons. Every non-comment statement must be terminated.
    """

    pending = ""
    # Check incrementally rather than line-by-line: migration authors may put
    # several complete statements on one line. SQLite still keeps a CREATE
    # TRIGGER body incomplete until its terminating END statement.
    for character in sql:
        pending += character
        if sqlite3.complete_statement(pending):
            if not _comment_only(pending):
                yield pending.strip()
            pending = ""
    if pending.strip() and not _comment_only(pending):
        raise ValueError("incomplete SQL migration statement; every statement must end with a semicolon")


def _comment_only(fragment: str) -> bool:
    fragment = re.sub(r"/\*.*?\*/", "", fragment, flags=re.DOTALL)
    fragment = re.sub(r"--[^\n]*(?:\n|$)", "", fragment)
    return not fragment.strip()
