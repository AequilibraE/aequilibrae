import ctypes
import logging
import os
import shutil
import urllib
import warnings
from os.path import basename, join
from pathlib import Path
from sqlite3 import Connection, OperationalError, register_adapter
from tempfile import gettempdir
from zipfile import ZipFile

import numpy as np

from aequilibrae.utils.db_utils import AequilibraEConnection, has_table, safe_connect
from aequilibrae.utils.qgis_utils import inside_qgis

# Setup adapters so that we can read/write numpy types directly to DB
register_adapter(np.int64, int)
register_adapter(np.int32, int)
register_adapter(np.float32, float)
register_adapter(np.float64, float)
register_adapter(object, str)


logger = logging.getLogger(__name__)


def is_windows():
    return os.name == "nt"


def is_not_windows():
    return os.name != "nt"


def connect_spatialite(path_to_file: os.PathLike | str, missing_ok: bool = False) -> Connection:
    if inside_qgis:
        import qgis

        return qgis.utils.spatialite_connect(str(path_to_file), factory=AequilibraEConnection)

    ensure_spatialite_binaries()

    return _connect_spatialite(path_to_file, missing_ok)


def _connect_spatialite(path_to_file: os.PathLike | str, missing_ok: bool = False):
    conn = safe_connect(path_to_file, missing_ok)
    load_spatialite_extension(conn)
    return conn


def load_spatialite_extension(conn: Connection):
    conn.enable_load_extension(True)
    directory = os.environ.get("AEQ_SPATIALITE_DIR")

    # Try loading from specific directory first
    if directory:
        try:
            _load_extension(conn, os.path.join(directory, "mod_spatialite"))
            return
        except OperationalError:
            logger.error(
                f"Environment variable 'AEQ_SPATIALITE_DIR' was provided ({directory}), "
                "but mod_spatialite could not be loaded from this directory. Trying system path"
            )

    try:
        _load_extension(conn, "mod_spatialite")
    except OperationalError as e:
        if is_windows():
            ensure_spatialite_binaries()
            try:
                # Retry after potential download
                directory = os.environ.get("AEQ_SPATIALITE_DIR", gettempdir())
                _load_extension(conn, os.path.join(directory, "mod_spatialite"))
                return
            except OperationalError as e2:
                raise e2 from e


_pinned_extensions: dict = {}


def _load_extension(conn: Connection, path: str) -> None:
    conn.load_extension(path)
    _pin_extension(path)


def _pin_extension(path: str) -> None:
    # SQLite loads mod_spatialite with LoadLibrary on every load_extension call and frees it with
    # FreeLibrary when the connection closes. On Windows, each load/unload cycle leaks a TLS index,
    # and the process aborts once the ~1088-slot limit is reached (~1000 connections). Holding one
    # extra reference here keeps the DLL permanently mapped so it is never actually unloaded.
    if is_not_windows() or path in _pinned_extensions:
        return
    try:
        # winmode=0 gives classic LoadLibrary search semantics (incl. PATH), matching how SQLite
        # itself resolves the extension. LoadLibrary appends ".dll" to the extension-less path,
        # resolving to the same module SQLite loaded.
        _pinned_extensions[path] = ctypes.CDLL(path, winmode=0)
    except OSError as e:
        logger.warning(f"Could not pin mod_spatialite ({path}) in memory: {e}")


def is_spatialite(conn):
    return has_table(conn, "geometry_columns")


def set_known_spatialite_folder(spatialite_folder: os.PathLike | str):
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
        logger.info(f"mod_spatialite.dll not found in {directory} attempting to download")
        try:
            _download_and_extract_spatialite(directory)
            os.environ["AEQ_SPATIALITE_DIR"] = directory
        except Exception as e:
            logger.error(f"Failed to download Spatialite binaries: {e}")
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
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)


def _dll_already_exists(d: os.PathLike | str) -> bool:
    return os.path.exists(join(d, "mod_spatialite.dll"))


def _download_and_extract_spatialite(directory: os.PathLike | str) -> None:
    url = "https://github.com/AequilibraE/aequilibrae/releases/download/v1.4.3/mod_spatialite-5.1.0-win-amd64.zip"
    zip_file = join(directory, basename(url))

    Path(directory).mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(url, zip_file)
    ZipFile(zip_file).extractall(directory)
    os.remove(zip_file)


def spatialize_db(conn, logger=None):
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
