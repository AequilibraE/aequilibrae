import logging
import os
from sqlite3 import Connection, register_adapter

import numpy as np

from aequilibrae.utils.db_utils import AequilibraEConnection, has_table, safe_connect
from aequilibrae.utils.qgis_utils import inside_qgis
from aequilibrae.utils.spatialite_shim import register_spatialite_functions, upgrade_legacy_spatialindex_triggers

# Setup adapters so that we can read/write numpy types directly to DB
for _type, _converter in ((np.int64, int), (np.int32, int), (np.float32, float), (np.float64, float), (object, str)):
    register_adapter(_type, _converter)


def connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False) -> Connection:
    if inside_qgis:
        import qgis

        return qgis.utils.spatialite_connect(str(path_to_file), factory=AequilibraEConnection)

    conn = safe_connect(path_to_file, missing_ok)
    load_spatialite_extension(conn)
    return conn


def _connect_spatialite(path_to_file: os.PathLike, missing_ok: bool = False):
    conn = safe_connect(path_to_file, missing_ok)
    load_spatialite_extension(conn)
    return conn


def load_spatialite_extension(conn: Connection):
    """Provide SpatiaLite-compatible spatial SQL on ``conn``.

    Registers pure-Python (shapely/pyproj-backed) implementations of the SpatiaLite
    functions AequilibraE uses. No native extension is loaded, so no system package
    or downloaded binary is required. Databases opened by projects created with
    older versions have their spatial-index triggers transparently upgraded.
    """
    register_spatialite_functions(conn)
    upgrade_legacy_spatialindex_triggers(conn)


def is_spatialite(conn):
    return has_table(conn, "geometry_columns")


def ensure_spatialite_binaries() -> None:
    """Deprecated no-op. AequilibraE no longer requires SpatiaLite binaries."""


def spatialize_db(conn, logger=None):
    logger = logger or logging.getLogger("aequilibrae")
    logger.info("Adding Spatialite infrastructure to the database")
    if not inside_qgis and not is_spatialite(conn):
        try:
            register_spatialite_functions(conn)
            conn.execute("SELECT InitSpatialMetaData();")
            conn.commit()
        except Exception as e:
            logger.error("Problem with spatialite", e.args)
            raise e
    if not is_spatialite(conn):
        raise RuntimeError("Something went wrong")
