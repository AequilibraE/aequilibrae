import sqlite3
import os
import shutil
from aequilibrae.project.network import Network
from aequilibrae.parameters import Parameters
from aequilibrae import logger
from aequilibrae.reference_files import spatialite_database
from .spatialite_connection import spatialite_connection
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
        self.source: str = None
        self.parameters = Parameters().parameters
        self.conn: sqlite3.Connection = None
        self.network: Network = None

    def load(self, project_path: str) -> None:
        """
        Loads project from disk

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If the project inside does
            not exist, it will fail.
        """
        file_name = os.path.join(project_path, 'project_database.sqlite')
        if not os.path.isfile(file_name):
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        self.project_base_path = project_path
        self.path_to_file = file_name
        self.source = self.path_to_file
        self.conn = sqlite3.connect(self.path_to_file)
        self.conn = spatialite_connection(self.conn)
        self.network = Network(self)
        os.environ['AEQUILIBRAE_PROJECT_PATH'] = self.project_base_path
        logger.info(f'Opened project on {self.project_base_path}')

    def new(self, project_path: str) -> None:
        """Creates a new project

        Args:
            *project_path* (:obj:`str`): Full path to the project data folder. If folder exists, it will fail
        """
        self.project_base_path = project_path
        self.path_to_file = os.path.join(self.project_base_path, 'project_database.sqlite')
        self.source = self.path_to_file

        if os.path.isdir(project_path):
            raise FileNotFoundError("Location already exists. Choose a different name or remove the existing directory")
        self.__create_empty_project()
        self.parameters = Parameters().parameters

        self.network = Network(self)
        logger.info(f'Created project on {self.project_base_path}')

    def close(self) -> None:
        """Safely closes the project"""
        self.conn.close()
        logger.info(f'Closed project on {self.project_base_path}')

    def __create_empty_project(self):

        # We create the project folder and create the base file
        os.mkdir(self.project_base_path)
        shutil.copyfile(spatialite_database, self.path_to_file)
        self.conn = spatialite_connection(sqlite3.connect(self.path_to_file))

        # We create the enviroment variable with the the location for the project
        os.environ['AEQUILIBRAE_PROJECT_PATH'] = self.project_base_path

        # Write parameters to the project folder
        p = Parameters()
        p.parameters["system"]["logging_directory"] = self.project_base_path
        p.write_back()

        # Create actual tables
        cursor = self.conn.cursor()
        cursor.execute('PRAGMA foreign_keys = ON;')
        self.conn.commit()
        initialize_tables(self.conn)
