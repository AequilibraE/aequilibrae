import os
import sqlite3
from aequilibrae.project.spatialite_connection import spatialite_connection
from aequilibrae.context import get_active_project
from aequilibrae.utils.qgis_utils import inside_qgis


def database_connection(project_path=None) -> sqlite3.Connection:
    project_path = project_path or get_active_project().project_base_path
    file_name = os.path.join(project_path, "project_database.sqlite")
    if not os.path.exists(file_name):
        raise FileExistsError

    if inside_qgis:
        import qgis

        conn = qgis.utils.spatialite_connect(file_name)
    else:
        conn = spatialite_connection(sqlite3.connect(file_name))
    return conn
