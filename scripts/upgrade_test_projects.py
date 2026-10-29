"""Upgrade all AequilibraE projects stored in test and reference data."""

import logging
import tempfile
import zipfile
from pathlib import Path

from aequilibrae.project.project import _upgrade
from aequilibrae.project.project_creation import remove_triggers
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.logging_utils import basic_config

ROOT_PATH = Path(__file__).resolve().parent.parent
PROJECT_DATABASE = "project_database.sqlite"
TRANSIT_DATABASE = "public_transport.sqlite"
RESULTS_DATABASE = "results_database.sqlite"
PROJECT_DATA_PATHS = (ROOT_PATH / "tests" / "data", ROOT_PATH / "src" / "aequilibrae" / "reference_files")


def upgrade_project(project_path: Path) -> None:
    """Open, upgrade, and close one project."""
    print(f"Upgrading {project_path}")
    db_path = project_path / "project_database.sqlite"
    transit_database_path = project_path / TRANSIT_DATABASE
    results_database_path = project_path / RESULTS_DATABASE

    transit_path = transit_database_path if transit_database_path.exists() else None
    results_path = results_database_path if results_database_path.exists() else None
    _upgrade(project_path=db_path, results_path=results_path, transit_path=transit_path)

    if project_path.stem == "no_triggers_project":
        with commit_and_close(db_path, spatial=True) as conn:
            remove_triggers(conn, "network")
        print("Removed triggers from no_triggers_project")


def upgrade_archive(archive_path: Path) -> None:
    """Upgrade projects in an archive and replace it with the upgraded archive."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not any(Path(member.filename).name == PROJECT_DATABASE for member in members):
                return
            comment = archive.comment
            archive.extractall(temporary_path)

        print(f"Upgrading projects in {archive_path}")
        for database_path in temporary_path.rglob(PROJECT_DATABASE):
            upgrade_project(database_path.parent)

        with tempfile.NamedTemporaryFile(dir=archive_path.parent, delete=False) as temporary_archive:
            temporary_archive_path = Path(temporary_archive.name)

        try:
            with zipfile.ZipFile(temporary_archive_path, "w") as upgraded_archive:
                upgraded_archive.comment = comment
                for member in members:
                    upgraded_archive.write(
                        temporary_path / member.filename,
                        arcname=member.filename,
                        compress_type=member.compress_type,
                    )
            temporary_archive_path.replace(archive_path)
        finally:
            temporary_archive_path.unlink(missing_ok=True)


def main() -> None:
    """Upgrade every unpacked or archived project in the configured data directories."""
    basic_config(level=logging.DEBUG)
    for data_path in PROJECT_DATA_PATHS:
        for database_path in sorted(data_path.rglob(PROJECT_DATABASE)):
            upgrade_project(database_path.parent)

        for archive_path in sorted(data_path.rglob("*.zip")):
            upgrade_archive(archive_path)


if __name__ == "__main__":
    main()
