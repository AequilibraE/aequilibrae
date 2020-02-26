import os
import platform
from aequilibrae.parameters import Parameters
from aequilibrae import logger


def spatialite_connection(conn):
    conn.enable_load_extension(True)
    plat = platform.platform()
    pth = os.getcwd()
    if "WINDOWS" in plat.upper():
        par = Parameters()
        spatialite_path = par.parameters["system"]["spatialite_path"]
        if os.path.isfile(os.path.join(spatialite_path, "mod_spatialite.dll")):
            os.chdir(spatialite_path)
    try:
        conn.load_extension("mod_spatialite")
    except Exception as e:
        logger.warn(f"AequilibraE might not work as intended without spatialite. {e.args}")
    os.chdir(pth)
    return conn
