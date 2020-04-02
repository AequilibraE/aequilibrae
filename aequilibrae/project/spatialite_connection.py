import os
from aequilibrae.parameters import Parameters
from aequilibrae import logger


def spatialite_connection(conn):
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
