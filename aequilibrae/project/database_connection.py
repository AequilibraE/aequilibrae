import os
import sqlite3
import importlib.util as iutil
from aequilibrae.project.spatialite_connection import spatialite_connection

spec = iutil.find_spec("qgis")
inside_qgis = spec is not None
if inside_qgis:
    import qgis

ENVIRON_VAR = "AEQUILIBRAE_PROJECT_PATH"


def database_connection() -> sqlite3.Connection:
    if ENVIRON_VAR not in os.environ:
        raise FileExistsError("There is no AequilibraE project loaded to connect to")

    path = os.environ.get(ENVIRON_VAR)
    file_name = os.path.join(path, "project_database.sqlite")
    if inside_qgis:
        conn = qgis.utils.spatialite_connect(file_name)
    else:
        conn = spatialite_connection(sqlite3.connect(file_name))
    return conn
