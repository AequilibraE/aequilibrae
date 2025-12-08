import sqlite3
from pathlib import Path

from aequilibrae.context import get_active_project
from aequilibrae.utils.spatialite_utils import connect_spatialite


def database_connection(db_type: str, project_path=None) -> sqlite3.Connection:
    return connect_spatialite(database_path(db_type, project_path))


def database_path(db_type: str, project_path=None) -> Path:
    project_path = project_path or get_active_project().project_base_path
    if db_type == "project" or db_type == "project_database" or db_type == "network":
        db = "project_database"
    elif db_type == "transit":
        db = "public_transport"
    elif db_type == "results":
        db = "results_database"
    else:
        raise ValueError(f"unknown database type {db_type}")

    return Path(project_path) / f"{db}.sqlite"
