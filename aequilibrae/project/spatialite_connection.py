import os
from sqlite3 import Connection


def spatialite_connection(conn: Connection) -> Connection:
    from aequilibrae.parameters import Parameters
    from aequilibrae import logger
    conn.enable_load_extension(True)
    par = Parameters()
    spatialite_path = par.parameters["system"]["spatialite_path"]
    if spatialite_path not in os.environ['PATH']:
        os.environ['PATH'] = spatialite_path + ';' + os.environ['PATH']
    try:
        conn.load_extension("mod_spatialite")
    except Exception as e:
        logger.warning(f"AequilibraE might not work as intended without spatialite. {e.args}")
    return conn
