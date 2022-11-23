import logging
import os
import shutil
import urllib
from os.path import join, basename
from pathlib import Path
from sqlite3 import Connection
from tempfile import gettempdir
from typing import Optional
from zipfile import ZipFile
from aequilibrae.utils.db_utils import has_table, safe_connect

inside_qgis = False

def is_windows():
    return os.name == "nt"


def is_not_windows():
    return os.name != "nt"


def connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False) -> Connection:
    if inside_qgis:
        import qgis

        return qgis.utils.spatialite_connect(path_to_file)

    ensure_spatialite_binaries()
    return _connect_spatialite(path_to_file, missing_ok)


def _connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False):
    conn = safe_connect(path_to_file, missing_ok)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    return conn


def is_spatialite(conn):
    return has_table(conn, "geometry_columns")


def ensure_spatialite_binaries(directory: Optional[os.PathLike] = None) -> None:
    if is_not_windows():
        return

    directory = directory or gettempdir()

    if not _dll_already_exists(directory):
        _download_and_extract_spatialite(directory)

    if directory not in os.environ["PATH"] or "PROJ_LIB" not in os.environ:
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        os.environ["PROJ_LIB"] = directory

    # We need to have the proj.db file in place.
    # The easiest one on Windows is in the public user. On Linux it should not be necessary
    # See why: https://www.gaia-gis.it/fossil/libspatialite/wiki?name=PROJ.6
    projdb_dir = "C:/Users/Public/spatialite/proj"
    Path(projdb_dir).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(join(projdb_dir, "proj.db")):
        return

    shutil.copyfile(join(directory, "proj.db"), join(projdb_dir, "proj.db"))


def _dll_already_exists(d: os.PathLike) -> bool:
    return os.path.exists(join(d, "mod_spatialite.dll"))


def _download_and_extract_spatialite(directory: os.PathLike) -> None:
    url = "https://polarisnetwork.s3.amazonaws.com/binaries/spatialite-loadable-modules-5.0.0-win-amd64.zip"
    zip_file = join(directory, basename(url))

    Path(directory).mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(url, zip_file)
    ZipFile(zip_file).extractall(directory)
    os.remove(zip_file)


def spatialize_db(conn, logger=None):
    logger = logging.getLogger("aequilibrae")
    logger.info("Adding Spatialite infrastructure to the database")
    curr = conn.cursor()
    if not inside_qgis and not is_spatialite(conn):
        try:
            curr.execute("SELECT InitSpatialMetaData();")
            conn.commit()
        except Exception as e:
            logger.error("Problem with spatialite", e.args)
            raise e
    if not is_spatialite(conn):
        raise RuntimeError("Something went wrong")