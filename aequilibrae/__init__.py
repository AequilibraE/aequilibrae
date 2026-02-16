import sys
from multiprocessing import set_start_method, get_start_method

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
from aequilibrae.project.project import Project
from aequilibrae.paths.results import AssignmentResults, SkimResults, PathResults

from aequilibrae import paths

__all__ = [
    "global_logger",
    "Parameters",
    "Matrices",
    "Log",
    "matrix",
    "transit",
    "project",
    "Ipf",
    "GravityApplication",
    "GravityCalibration",
    "SyntheticGravityModel",
    "AequilibraeMatrix",
    "distribution",
    "NetworkSkimming",
    "TrafficClass",
    "VDF",
    "allOrNothing",
    "TrafficAssignment",
    "Graph",
    "AssignmentPaths",
    "Project",
    "AssignmentResults",
    "SkimResults",
    "PathResults",
    "paths",
]

# When updating the version, one must also update the docs/source/useful_links/version_history.rst file
version = "1.5.2"

# On macos, we start multiprocessing with 'fork' to avoid segfaults. Other platform defaults are fine
if sys.platform == "darwin" and get_start_method(allow_none=True) != "fork":
    try:
        set_start_method("fork")
    except RuntimeError:
        logger.critical(
            "multiprocessing start method already set. On MacOS, AequilibraE requires the 'fork' start method. "
            "AequilibraE may crash when using procedures that utilise multiprocessing or progress bars."
        )
