from aequilibrae.starts_logging import logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.data import Matrices
from aequilibrae.log import Log
from aequilibrae import distribution
from aequilibrae import matrix
from aequilibrae import transit
from aequilibrae import project
import warnings

compiled = True
try:
    from aequilibrae.paths.AoN import path_computation
except Exception as e:
    compiled = False
    warnings.warn(f"Failed to import compiled modules. {e.args}")

if compiled:
    from aequilibrae.distribution import Ipf, GravityApplication, GravityCalibration, SyntheticGravityModel
    from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData

    from aequilibrae.paths.network_skimming import NetworkSkimming
    from aequilibrae.paths.traffic_class import TrafficClass
    from aequilibrae.paths.vdf import VDF
    from aequilibrae.paths.all_or_nothing import allOrNothing
    from aequilibrae.paths.traffic_assignment import TrafficAssignment
    from aequilibrae.paths.graph import Graph
    from aequilibrae.paths.select_link import SelectLink
    from aequilibrae.project import Project
    from aequilibrae.transit.gtfs import GTFS, create_gtfsdb
    from aequilibrae.paths.results import AssignmentResults, SkimResults, PathResults
    from aequilibrae.paths import release_version as __version__

    from aequilibrae import paths
name = "aequilibrae"
