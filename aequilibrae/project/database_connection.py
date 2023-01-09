import os
from os.path import join
import sqlite3
import importlib.util as iutil
from aequilibrae.project.spatialite_connection import spatialite_connection
from aequilibrae.context import get_active_project

spec = iutil.find_spec("qgis")
inside_qgis = spec is not None


def database_connection(table_type: str, project_path=None) -> sqlite3.Connection:
    project_path = project_path or get_active_project().project_base_path
    data_name = "project_database.sqlite" if table_type == "network" else "public_transport.sqlite"
    file_name = join(project_path, data_name)
    if not os.path.exists(file_name):
        raise FileExistsError
    if inside_qgis:
        import qgis

        return qgis.utils.spatialite_connect(file_name)
    else:
        return spatialite_connection(sqlite3.connect(file_name))
