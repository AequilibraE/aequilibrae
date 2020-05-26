import os
import sqlite3
from aequilibrae.project.spatialite_connection import spatialite_connection

environ_var = 'AEQUILIBRAE_PROJECT_PATH'


def database_connection() -> sqlite3.Connection:
    if environ_var not in os.environ:
        raise FileExistsError("There is no AequilibraE project loaded to connect to")

    path = os.environ.get(environ_var)
    file_name = os.path.join(path, 'project_database.sqlite')
    conn = spatialite_connection(sqlite3.connect(file_name))
    return conn
