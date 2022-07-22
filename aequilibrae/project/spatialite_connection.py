import os
from sqlite3 import Connection
import sqlite3
import numpy as np

sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.object0, str)


def spatialite_connection(conn: Connection) -> Connection:
    from aequilibrae.parameters import Parameters
    from aequilibrae import global_logger

    conn.enable_load_extension(True)
    par = Parameters()
    spatialite_path = par.parameters["system"]["spatialite_path"]
    if spatialite_path not in os.environ["PATH"]:
        os.environ["PATH"] = spatialite_path + ";" + os.environ["PATH"]
    try:
        conn.load_extension("mod_spatialite")
    except Exception as e:
        global_logger.warning(f"AequilibraE might not work as intended without spatialite. {e.args}")
    return conn
