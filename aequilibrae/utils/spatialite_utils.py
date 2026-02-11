import logging
import os
import shutil
import urllib
import warnings
from os.path import basename, join
from pathlib import Path
from sqlite3 import Connection, register_adapter, OperationalError
from tempfile import gettempdir
from typing import Optional
from zipfile import ZipFile

import numpy as np

from aequilibrae.log import global_logger
from aequilibrae.utils.db_utils import AequilibraEConnection, has_table, safe_connect
from aequilibrae.utils.qgis_utils import inside_qgis

# Setup adapters so that we can read/write numpy types directly to DB
register_adapter(np.int64, int)
register_adapter(np.int32, int)
register_adapter(np.float32, float)
register_adapter(np.float64, float)
register_adapter(object, str)


def is_windows():
    return os.name == "nt"


def is_not_windows():
    return os.name != "nt"


def connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False) -> Connection:
    if inside_qgis:
        import qgis

        return qgis.utils.spatialite_connect(str(path_to_file), factory=AequilibraEConnection)

    ensure_spatialite_binaries()

    return _connect_spatialite(path_to_file, missing_ok)


def _connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False):
    conn = safe_connect(path_to_file, missing_ok)
    load_spatialite_extension(conn)
    return conn


def load_spatialite_extension(conn: Connection):
    conn.enable_load_extension(True)
    directory = os.environ.get("AEQ_SPATIALITE_DIR")

    # Try loading from specific directory first
    if directory:
        try:
            conn.load_extension(os.path.join(directory, "mod_spatialite"))
            return
        except OperationalError:
            global_logger.error(
                f"Environment variable 'AEQ_SPATIALITE_DIR' was provided ({directory}), "
                "but mod_spatialite could not be loaded from this directory. Trying system path"
            )

    try:
        conn.load_extension("mod_spatialite")
    except OperationalError as e:
        if is_windows():
            ensure_spatialite_binaries()
            try:
                # Retry after potential download
                directory = os.environ.get("AEQ_SPATIALITE_DIR", gettempdir())
                conn.load_extension(os.path.join(directory, "mod_spatialite"))
                return
            except OperationalError as e2:
                raise e2 from e


def is_spatialite(conn):
    return has_table(conn, "geometry_columns")


def set_known_spatialite_folder(spatialite_folder: os.PathLike):
    directory = str(spatialite_folder)
    if directory not in os.environ["PATH"]:
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
    if "PROJ_LIB" not in os.environ:
        os.environ["PROJ_LIB"] = directory


def ensure_spatialite_binaries() -> None:
    if is_not_windows():
        return

    directory = os.environ.get("AEQ_SPATIALITE_DIR", gettempdir())

    if not _dll_already_exists(directory):
        global_logger.info(f"mod_spatialite.dll not found in {directory} attempting to download")
        try:
            _download_and_extract_spatialite(directory)
            os.environ["AEQ_SPATIALITE_DIR"] = directory
        except Exception as e:
            global_logger.error(f"Failed to download Spatialite binaries: {e}")
            raise e

    set_known_spatialite_folder(directory)

    try:
        # We need to have the proj.db file in place.
        # The easiest one on Windows is in the public user. On Linux it should not be necessary
        # See why: https://www.gaia-gis.it/fossil/libspatialite/wiki?name=PROJ.6
        projdb_dir = "C:/Users/Public/spatialite/proj"
        Path(projdb_dir).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(join(projdb_dir, "proj.db")):
            return

        shutil.copyfile(join(directory, "proj.db"), join(projdb_dir, "proj.db"))
    except Exception as e:
        msg = f"Could not put the proj.db file in the expected place. {e.args}"
        warnings.warn(msg)
        global_logger.warning(msg)


def _dll_already_exists(d: os.PathLike) -> bool:
    return os.path.exists(join(d, "mod_spatialite.dll"))


def _download_and_extract_spatialite(directory: os.PathLike) -> None:
    url = "https://github.com/AequilibraE/aequilibrae/releases/download/v1.4.3/mod_spatialite-5.1.0-win-amd64.zip"
    zip_file = join(directory, basename(url))

    Path(directory).mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(url, zip_file)
    ZipFile(zip_file).extractall(directory)
    os.remove(zip_file)


def spatialize_db(conn, logger=None):
    logger = logger or logging.getLogger("aequilibrae")
    logger.info("Adding Spatialite infrastructure to the database")
    if not inside_qgis and not is_spatialite(conn):
        try:
            conn.execute("SELECT InitSpatialMetaData();")
            conn.commit()
        except Exception as e:
            logger.error("Problem with spatialite", e.args)
            raise e
    if not is_spatialite(conn):
        raise RuntimeError("Something went wrong")
