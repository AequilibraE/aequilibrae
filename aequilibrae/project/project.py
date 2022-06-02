import logging
import os
import shutil
import sqlite3
import warnings

from aequilibrae import logger
from aequilibrae.log import Log
from aequilibrae.parameters import Parameters
from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices
from aequilibrae.project.database_connection import database_connection
from aequilibrae.context import activate_project, get_active_project
from aequilibrae.project.network import Network
from aequilibrae.project.zoning import Zoning
from aequilibrae.reference_files import spatialite_database
from aequilibrae.starts_logging import StartsLogging
from .project_cleaning import clean
from .project_creation import initialize_tables


class Project:
    """AequilibraE project class

    ::

        from aequilibrae.project import Project

        existing = Project()
        existing.load('path/to/existing/project/folder')

        newfile = Project()
        newfile.new('path/to/new/project/folder')
    """

    def __init__(self):
        self.path_to_file: str = None
        self.project_base_path = ""
        self.source: str = None
        self.conn: sqlite3.Connection = None
        self.network: Network = None
        self.about: About = None
        self.logger: logging.Logger = None

    def open(self, project_path: str) -> None:
        """
        Loads project from disk

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If the project inside does
            not exist, it will fail.
        """

        file_name = os.path.join(project_path, "project_database.sqlite")
        if not os.path.isfile(file_name):
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        self.project_base_path = project_path
        self.path_to_file = file_name
        self.source = self.path_to_file
        self.conn = database_connection(self.project_base_path)

        self.__load_objects()
        self.__set_logging_path()
        logger.info(f"Opened project on {self.project_base_path}")
        self.logger = logger
        clean(self.project_base_path)
        self.activate()

    def new(self, project_path: str) -> None:
        """Creates a new project

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If folder exists, it will fail
        """

        self.project_base_path = project_path
        self.path_to_file = os.path.join(self.project_base_path, "project_database.sqlite")
        self.source = self.path_to_file

        if os.path.isdir(project_path):
            raise FileNotFoundError("Location already exists. Choose a different name or remove the existing directory")
        self.activate()

        self.__create_empty_project()
        self.__load_objects()
        self.about.create()
        self.__set_logging_path()
        self.logger = logger
        logger.info(f"Created project on {self.project_base_path}")

    def close(self) -> None:
        """Safely closes the project"""
        if not self.project_base_path:
            warnings.warn("This Aequilibrae project is not opened")
            return
        try:
            self.conn.commit()
            clean(self.project_base_path)
            self.conn.close()
            for obj in [self.parameters, self.network]:
                del obj

            del self.network.link_types
            del self.network.modes

        except (sqlite3.ProgrammingError, AttributeError):
            warnings.warn(f"This project at {self.project_base_path} is already closed")
        else:
            logger.info(f"Closed project on {self.project_base_path}")
        finally:
            self.deactivate()

    def load(self, project_path: str) -> None:
        """
        Loads project from disk

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If the project inside does
            not exist, it will fail.
        """
        warnings.warn(f"Function has been deprecated. Use my_project.open({project_path}) instead", DeprecationWarning)
        self.open(project_path)

    def activate(self):
        activate_project(self)

    def deactivate(self):
        if get_active_project(must_exist=False) is self:
            activate_project(None)

    def log(self) -> Log:
        """Returns a log object

        allows the user to read the log or clear it"""
        return Log(self.project_base_path)

    def __load_objects(self):
        matrix_folder = os.path.join(self.project_base_path, "matrices")
        if not os.path.isdir(matrix_folder):
            os.mkdir(matrix_folder)

        self.network = Network(self)
        self.about = About(self.conn)
        self.matrices = Matrices(self)

    @property
    def project_parameters(self) -> Parameters:
        return Parameters(self.project_base_path)

    @property
    def parameters(self) -> dict:
        return self.project_parameters.parameters

    def check_file_indices(self) -> None:
        """Makes results_database.sqlite and the matrices folder compatible with project database"""
        raise NotImplementedError

    @property
    def zoning(self):
        return Zoning(self.network)

    def __create_empty_project(self):

        # We create the project folder and create the base file
        os.mkdir(self.project_base_path)
        shutil.copyfile(spatialite_database, self.path_to_file)

        self.conn = database_connection(self.project_base_path)

        # Write parameters to the project folder
        p = self.project_parameters
        p.parameters["system"]["logging_directory"] = self.project_base_path
        p.write_back()
        StartsLogging(self.project_base_path)

        # Create actual tables
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        self.conn.commit()
        initialize_tables(self.conn)

    def __set_logging_path(self):
        p = self.project_parameters
        par = p.parameters
        if p.parameters is None:
            par = p._default
        do_log = par["system"]["logging"]
        for handler in logger.handlers:
            if handler.name == "aequilibrae":
                logger.removeHandler(handler)
        if do_log:
            formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")
            log_file = os.path.join(self.project_base_path, "aequilibrae.log")
            if not os.path.isfile(log_file):
                a = open(log_file, "w")
                a.close()
            ch = logging.FileHandler(log_file)
            ch.name = "aequilibrae"
            ch.setFormatter(formatter)
            ch.setLevel(logging.DEBUG)
            logger.addHandler(ch)
