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

        existing = Project('path/to/existing/project.sqlite')

        newfile = Project('path/to/new/project.sqlite', True)
        """

    def __init__(self, path_to_file: str, new_project=False):
        """
        Instantiates the class by opening an existing project or creating a new one

        Args:
            *path_to_file* (:obj:`str`): Full path to the project data file. If project does not exist, new project
                                        argument needs to be True

            *new_project* (:obj:`bool`, Optional): Flag to create new project. *path_to_file* needs to be set to a
                                                   non-existing file.
        """
        self.path_to_file = path_to_file
        self.parameters = Parameters().parameters
        if not os.path.isfile(path_to_file):
            if not new_project:
                raise FileNotFoundError(
                    "Model does not exist. Check your path or use the new_project=True flag to create a new project"
                )
            else:
                self.__create_empty_project()
        else:
            self.conn = sqlite3.connect(self.path_to_file)

        self.source = self.path_to_file
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
