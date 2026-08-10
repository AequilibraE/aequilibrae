import importlib.metadata
import logging

from aequilibrae import distribution, matrix, paths, project, transit
from aequilibrae.distribution import GravityApplication, GravityCalibration, Ipf, SyntheticGravityModel
from aequilibrae.log import Log
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.assignment_paths import AssignmentPaths
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import AssignmentResults, PathResults, SkimResults
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.vdf import VDF
from aequilibrae.project.data import Matrices
from aequilibrae.project.project import Project

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

logger = global_logger = logging.getLogger(__name__)

version = importlib.metadata.version("aequilibrae")
