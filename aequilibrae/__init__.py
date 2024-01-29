import tempfile
import os
import glob
import sys
from aequilibrae.log import global_logger

try:
    pass
except Exception as e:
    global_logger.warning(f"Failed to import compiled modules. {e.args}")
    raise



name = "aequilibrae"


def setup():
    sys.dont_write_bytecode = True
    cleaning()


def cleaning():
    p = tempfile.gettempdir() + "/aequilibrae_*"
    for f in glob.glob(p):
        try:
            os.unlink(f)
        except Exception as err:
            global_logger.warning(err.__str__())


setup()
