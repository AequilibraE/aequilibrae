from os.path import join
import sqlite3
from aequilibrae.context import get_active_project
from aequilibrae.utils.spatialite_utils import connect_spatialite


def database_connection(table_type: str, project_path=None) -> sqlite3.Connection:
    project_path = project_path or get_active_project().project_base_path
    return connect_spatialite(join(project_path, "project_database.sqlite"))
