import sqlite3
import os
import shutil
from aequilibrae.project.network import Network
from aequilibrae.parameters import Parameters
from aequilibrae.reference_files import spatialite_database
from .spatialite_connection import spatialite_connection


class Project:
    """AequilibraE project class

    ::

        from aequilibrae.project import Project

        existing = Project()
        existing.load('path/to/existing/project.sqlite')

        newfile = Project()
        newfile.new('path/to/new/project.sqlite')
        """

    def __init__(self):
        self.path_to_file: str = None
        self.source: str = None
        self.parameters = Parameters().parameters
        self.conn: sqlite3.Connection = None
        self.network: Network = None

    def load(self, file_name: str) -> None:
        """
        Loads project from disk

        Args:
            *file_name* (:obj:`str`): Full path to the project data file. If does not exist, it will fail
        """
        self.path_to_file = file_name
        if not os.path.isfile(file_name):
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        self.conn = sqlite3.connect(self.path_to_file)
        self.source = self.path_to_file
        self.conn = spatialite_connection(self.conn)
        self.network = Network(self)

    def new(self, file_name: str) -> None:
        """Creates a new project

        Args:
            *file_name* (:obj:`str`): Full path to the project data file. If file exists, it will fail
        """
        self.path_to_file = file_name
        self.source = self.path_to_file
        self.parameters = Parameters().parameters
        if os.path.isfile(file_name):
            raise FileNotFoundError("File already exist. Choose a different name or remove the existing file")
        self.__create_empty_project()

        self.conn = spatialite_connection(self.conn)
        self.network = Network(self)

    def __create_empty_project(self):
        shutil.copyfile(spatialite_database, self.path_to_file)
        self.conn = sqlite3.connect(self.path_to_file)
        self.__create_modes_table()

    def __create_modes_table(self):

        create_query = """CREATE TABLE 'modes' (mode_name VARCHAR UNIQUE NOT NULL,
                                                mode_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                description VARCHAR);"""
        cursor = self.conn.cursor()
        cursor.execute(create_query)
        modes = self.parameters["network"]["modes"]

        for mode in modes:
            nm = list(mode.keys())[0]
            descr = mode[nm]["description"]
            mode_id = mode[nm]["letter"]
            par = [f'"{p}"' for p in [nm, mode_id, descr]]
            par = ",".join(par)
            sql = f"INSERT INTO 'modes' (mode_name, mode_id, description) VALUES({par})"
            cursor.execute(sql)
        self.conn.commit()
