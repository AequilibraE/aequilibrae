import tempfile
import os
import glob
import sys
from aequilibrae.log import logger, global_logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.data import Matrices
from aequilibrae.log import Log
from aequilibrae import matrix
from aequilibrae import transit
from aequilibrae import project

try:
    from aequilibrae.paths.AoN import path_computation
except Exception as e:
    global_logger.warning(f"Failed to import compiled modules. {e.args}")
    raise

from aequilibrae.distribution import Ipf, GravityApplication, GravityCalibration, SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
from aequilibrae import distribution
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.vdf import VDF
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.assignment_paths import AssignmentPaths
from aequilibrae.project import Project
from aequilibrae.paths.results import AssignmentResults, SkimResults, PathResults

from aequilibrae import paths

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
