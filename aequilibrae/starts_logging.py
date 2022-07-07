import os
import sys
import tempfile
import logging

from aequilibrae.context import get_active_project
from .parameters import Parameters
import glob

sys.dont_write_bytecode = True


# TODO: Add tests for logging
def StartsLogging(project=None):
    # CREATE THE LOGGER
    project = project or get_active_project(must_exist=False)
    project_path = project.project_base_path if project is not None else ""

    p = Parameters(project_path)
    par = p.parameters
    if p.parameters is None:
        par = p._default
    temp_folder = par["system"]["logging_directory"]
    do_log = par["system"]["logging"]
    if not os.path.isdir(temp_folder):
        temp_folder = tempfile.gettempdir()

    if do_log:
        log_file = os.path.join(temp_folder, "aequilibrae.log")
        if not os.path.isfile(log_file):
            a = open(log_file, "w")
            a.close()

    logger = logging.getLogger("aequilibrae")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")

    if not len(logger.handlers):
        if do_log:
            ch = logging.FileHandler(log_file)
        ch.setFormatter(formatter)
        ch.name = "aequilibrae"
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    return logger


def cleaning():
    p = tempfile.gettempdir() + "/aequilibrae_*"
    for f in glob.glob(p):
        try:
            os.unlink(f)
        except Exception as err:
            logger.warning(err.__str__())


logger = StartsLogging()
cleaning()
