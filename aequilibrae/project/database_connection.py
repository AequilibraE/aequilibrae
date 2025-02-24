import sqlite3
from pathlib import Path

from aequilibrae.context import get_active_project
from aequilibrae.utils.spatialite_utils import connect_spatialite


def database_connection(db_type: str, project_path=None) -> sqlite3.Connection:
    return connect_spatialite(database_path(db_type, project_path))


def database_path(db_type: str, project_path=None) -> Path:
    project_path = project_path or get_active_project().project_base_path
    db = "public_transport" if db_type == "transit" else "project_database"
    return Path(project_path) / f"{db}.sqlite"
