import os
import sys
import tempfile
import logging

from aequilibrae.context import get_active_project
from .parameters import Parameters
import glob

sys.dont_write_bytecode = True


# TODO: Add tests for logging
def StartsLogging():
    # CREATE THE GLOBAL LOGGER

    par = Parameters._default
    temp_folder = par["system"]["logging_directory"]
    do_log = par["system"]["logging"]
    if not os.path.isdir(temp_folder):
        temp_folder = tempfile.gettempdir()

    logger = logging.getLogger("aequilibrae")
    logger.setLevel(logging.DEBUG)

    if not len(logger.handlers) and do_log:
        log_file = os.path.join(temp_folder, "aequilibrae.log")
        logger.addHandler(get_log_handler(log_file))
    return logger


def get_log_handler(log_file: str, ensure_file_exists=True):
    """return a log handler that writes to the given log_file"""
    if os.path.exists(log_file) and not os.path.isfile(log_file):
        raise FileExistsError(f"{log_file} is not a valid file")

    if ensure_file_exists:
        open(log_file, "a").close()

    formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.name = "aequilibrae"
    handler.setLevel(logging.DEBUG)
    return handler


def cleaning():
    p = tempfile.gettempdir() + "/aequilibrae_*"
    for f in glob.glob(p):
        try:
            os.unlink(f)
        except Exception as err:
            global_logger.warning(err.__str__())


global_logger = logger = StartsLogging()
cleaning()
