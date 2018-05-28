"""
=============
Core AequilibraE
=============

Imports AequilibraE modules

"""
import sys
sys.dont_write_bytecode = True

from . import paths
from . import distribution
from . import matrix
from . import utils
from . import transit
import reserved_fields
from parameters import Parameters
from .reference_files import spatialite_database


from parameters import Parameters
import logging
import tempfile
import os

# CREATE THE LOGGER
temp_folder = Parameters().parameters['system']['temp directory']
if not os.path.isdir(temp_folder):
    temp_folder = tempfile.gettempdir()

log_file = os.path.join(temp_folder, 'aequilibrae.log')
if not os.path.isfile(log_file):
    a = open(log_file, 'w')
    a.close()

logger = logging.getLogger('aequilibrae')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not len(logger.handlers):
    ch = logging.FileHandler(log_file)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
