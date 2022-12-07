import os
import sqlite3
import importlib.util as iutil
from aequilibrae.context import get_active_project
from aequilibrae.project.spatialite_connection import spatialite_connection

spec = iutil.find_spec("qgis")
inside_qgis = spec is not None
if inside_qgis:
    import qgis


def transit_connection(project_path=None) -> sqlite3.Connection:
    project_path = project_path or get_active_project().project_base_path
    file_name = os.path.join(project_path, "public_transport.sqlite")
    if not os.path.exists(file_name):
        raise FileExistsError
    if inside_qgis:
        conn = qgis.utils.spatialite_connect(file_name)
    else:
        conn = spatialite_connection(sqlite3.connect(file_name))
    return conn
