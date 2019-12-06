import os
import sys
import tempfile
import logging
from .parameters import Parameters

sys.dont_write_bytecode = True


def StartsLogging():
    # CREATE THE LOGGER
    temp_folder = Parameters().parameters["system"]["temp directory"]
    if not os.path.isdir(temp_folder):
        temp_folder = tempfile.gettempdir()

    log_file = os.path.join(temp_folder, "aequilibrae.log")
    if not os.path.isfile(log_file):
        a = open(log_file, "w")
        a.close()

    logger = logging.getLogger("aequilibrae")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not len(logger.handlers):
        ch = logging.FileHandler(log_file)
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
