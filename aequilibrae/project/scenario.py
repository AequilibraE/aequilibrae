import pathlib
import logging

from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network


class Scenario:
    """
    Represents a modelling scenario within an AequilibraE project.

    Each scenario operates independently with its own database and file
    structure while sharing the overall project configuration.

    Scenarios are typically managed through the Project class rather than
    instantiated directly by users.

    The root scenario is special-cased and represents the original project
    configuration. All other scenarios are stored in subdirectories and
    reference their own database files.
    """

    name: str
    base_path: pathlib.Path
    path_to_file: pathlib.Path
    logger: logging.Logger

    about: About
    network: Network
    matrices: Matrices
    results: Results

    def __init__(self, name: str, base_path: pathlib.Path, path_to_file: pathlib.Path):
        self.name = name
        self.base_path = base_path
        self.path_to_file = path_to_file
