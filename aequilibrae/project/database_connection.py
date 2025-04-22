import sqlite3
from os.path import join

from aequilibrae.context import get_active_project
from aequilibrae.utils.spatialite_utils import connect_spatialite


def database_connection(db_type: str, project_path=None) -> sqlite3.Connection:
    return connect_spatialite(database_path(db_type, project_path))


def database_path(db_type: str, project_path=None):
    project_path = project_path or get_active_project().project_base_path
    db = "public_transport" if db_type == "transit" else "project_database"
    return join(project_path, f"{db}.sqlite")
