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
from aequilibrae.project.database_connection import database_connection, ENVIRON_VAR
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

        if self.__other_project_still_open():
            raise Exception("You already have a project open. Close that project before opening another one")

        file_name = os.path.join(project_path, "project_database.sqlite")
        if not os.path.isfile(file_name):
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        self.project_base_path = project_path
        self.path_to_file = file_name
        self.source = self.path_to_file
        os.environ[ENVIRON_VAR] = self.project_base_path
        self.conn = database_connection()

        self.__load_objects()
        self.__set_logging_path()
        logger.info(f"Opened project on {self.project_base_path}")
        self.logger = logger
        clean()

    def new(self, project_path: str) -> None:
        """Creates a new project

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If folder exists, it will fail
        """
        if self.__other_project_still_open():
            raise Exception("You already have a project open. Close that project before creating a new one")

        self.project_base_path = project_path
        self.path_to_file = os.path.join(self.project_base_path, "project_database.sqlite")
        self.source = self.path_to_file

        if os.path.isdir(project_path):
            raise FileNotFoundError("Location already exists. Choose a different name or remove the existing directory")
        os.environ[ENVIRON_VAR] = self.project_base_path

        self.__create_empty_project()
        self.__load_objects()
        self.about.create()
        self.__set_logging_path()
        self.logger = logger
        logger.info(f"Created project on {self.project_base_path}")

    def close(self) -> None:
        """Safely closes the project"""
        if ENVIRON_VAR in os.environ:
            self.conn.commit()
            clean()
            self.conn.close()
            for obj in [self.parameters, self.network]:
                del obj
            del os.environ[ENVIRON_VAR]
            del self.network.link_types
            del self.network.modes
            logger.info(f"Closed project on {self.project_base_path}")
        else:
            warnings.warn("There is no Aequilibrae project open that you could close")

    def load(self, project_path: str) -> None:
        """
        Loads project from disk

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If the project inside does
            not exist, it will fail.
        """
        warnings.warn(f"Function has been deprecated. Use my_project.open({project_path}) instead", DeprecationWarning)
        self.open(project_path)

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

    @property
    def matrices(self) -> Matrices:
        return Matrices()

    @property
    def parameters(self) -> dict:
        return Parameters().parameters

    def check_file_indices(self) -> None:
        """ Makes results_database.sqlite and the matrices folder compatible with project database
        """
        raise NotImplementedError

    @property
    def zoning(self):
        return Zoning(self.network)

    def __create_empty_project(self):

        # We create the project folder and create the base file
        os.mkdir(self.project_base_path)
        shutil.copyfile(spatialite_database, self.path_to_file)

        self.conn = database_connection()

        # Write parameters to the project folder
        p = Parameters()
        p.parameters["system"]["logging_directory"] = self.project_base_path
        p.write_back()
        _ = StartsLogging()

        # Create actual tables
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        self.conn.commit()
        initialize_tables(self.conn)

    def __other_project_still_open(self) -> bool:
        if ENVIRON_VAR in os.environ:
            return True
        return False

    def __set_logging_path(self):
        p = Parameters()
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
