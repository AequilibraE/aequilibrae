from aequilibrae.log import logger, global_logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.data import Matrices
from aequilibrae.log import Log
from aequilibrae import matrix
from aequilibrae import transit
from aequilibrae import project

from aequilibrae.distribution import Ipf, GravityApplication, GravityCalibration, SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix
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

from multiprocessing import set_start_method
import sys

# When updating the version, one must also update the docs/source/useful_links/version_history.rst file
version = "1.5.0"

# On macos, we start multiprocessing with 'fork' to avoid segfaults. Other platform defaults are fine
if sys.platform == "darwin":
    try:
        set_start_method("fork")
    except RuntimeError:
        # start method has already been set
        pass