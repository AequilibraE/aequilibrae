import pathlib
import logging

from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network


class Scenario:
    base_path: pathlib.Path
    path_to_file: pathlib.Path
    logger: logging.Logger

    about: About
    network: Network
    matrices: Matrices
    results: Results

    def __init__(self, base_path: pathlib.Path, path_to_file: pathlib.Path):
        self.base_path = base_path
        self.path_to_file = path_to_file
