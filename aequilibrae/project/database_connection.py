import os
import sqlite3
from aequilibrae.project.spatialite_connection import spatialite_connection

ENVIRON_VAR = 'AEQUILIBRAE_PROJECT_PATH'


def database_connection() -> sqlite3.Connection:
    if ENVIRON_VAR not in os.environ:
        raise FileExistsError("There is no AequilibraE project loaded to connect to")

    path = os.environ.get(ENVIRON_VAR)
    file_name = os.path.join(path, 'project_database.sqlite')
    conn = spatialite_connection(sqlite3.connect(file_name))
    return conn
