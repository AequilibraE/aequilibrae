import sqlite3
import os
import shutil
from aequilibrae.project.network import Network
from aequilibrae.parameters import Parameters
from aequilibrae.reference_files import spatialite_database
from .spatialite_connection import spatialite_connection

meta_table = 'attributes_documentation'


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

    def close(self) -> None:
        """Safely closes the project"""
        self.conn.close()

    def __create_empty_project(self):
        shutil.copyfile(spatialite_database, self.path_to_file)
        self.conn = sqlite3.connect(self.path_to_file)

        cursor = self.conn.cursor()
        cursor.execute('PRAGMA foreign_keys = ON;')
        self.conn.commit()
        self.__create_meta_table()
        self.__create_modes_table()
        self.__create_link_type_table()

    def __create_meta_table(self):
        cursor = self.conn.cursor()
        create_query = f"""CREATE TABLE '{meta_table}' (name_table  VARCHAR UNIQUE NOT NULL,
                                                        link_type_attribute VARCHAR UNIQUE NOT NULL,
                                                        description VARCHAR);"""
        cursor.execute(create_query)
        self.conn.commit()

    def __create_modes_table(self):

        create_query = """CREATE TABLE 'modes' (mode_name VARCHAR UNIQUE NOT NULL,
                                                mode_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                description VARCHAR,
                                                alpha NUMERIC,
                                                beta NUMERIC,
                                                gamma NUMERIC,
                                                delta NUMERIC,
                                                epsilon NUMERIC,
                                                zeta NUMERIC,
                                                iota NUMERIC,
                                                sigma NUMERIC,
                                                phi NUMERIC,
                                                tau NUMERIC);"""
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

    def __create_link_type_table(self):

        create_query = """CREATE TABLE 'link_types' (link_type VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                     link_type_id VARCHAR UNIQUE NOT NULL,
                                                     description VARCHAR,
                                                     lanes NUMERIC,
                                                     lane_capacity NUMERIC,
                                                     alpha NUMERIC,
                                                     beta NUMERIC,
                                                     gamma NUMERIC,
                                                     delta NUMERIC,
                                                     epsilon NUMERIC,
                                                     zeta NUMERIC,
                                                     iota NUMERIC,
                                                     sigma NUMERIC,
                                                     phi NUMERIC,
                                                     tau NUMERIC);"""

        cursor = self.conn.cursor()
        cursor.execute(create_query)

        link_types = self.parameters["network"]["links"]["link_types"]
        sql = "INSERT INTO 'link_types' (link_type, link_type_id, description, lanes, lane_capacity) VALUES(?, ?, ?, ?, ?)"
        for lt in link_types:
            nm = list(lt.keys())[0]
            args = (nm, lt[nm]["link_type_id"], lt[nm]["description"], lt[nm]["lanes"], lt[nm]["lane_capacity"])

            cursor.execute(sql, args)

        self.conn.commit()

    def __add_meta_extra_attributes(self):
        fields = []
        fields.append(['link_types', 'link_type', 'Link type name. E.g. arterial, or connector'])
        fields.append(['link_types', 'link_type_id', 'Single letter identifying the mode. E.g. a, for arterial'])
        fields.append(['link_types', 'description', 'Description of the same. E.g. Arterials are streets like AequilibraE Avenue'])
        fields.append(['link_types', 'lanes', 'Default number of lanes in each direction. E.g. 2'])
        fields.append(['link_types', 'lane_capacity', 'Default vehicle capacity per lane. E.g.  900'])

        fields.append(['modes', 'mode_name', 'Link type name. E.g. arterial, or connector'])
        fields.append(['modes', 'mode_id', 'Single letter identifying the mode. E.g. a, for arterial'])
        fields.append(['modes', 'description', 'Description of the same. E.g. Arterials are streets like AequilibraE Avenue'])

        extra_keys = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'iota', 'sigma', 'phi', 'tau']
        extra_keys = [[x, 'Available for user convenience'] for x in extra_keys]

        cursor = self.conn.cursor()
        for table_name, f, d in fields:
            sql = f"INSERT INTO '{meta_table}' (name_table, attribute, description) VALUES('{table_name}','{f}', '{d}')"
            cursor.execute(sql)

        for table_name in ['link_types', 'modes']:
            for f, d in extra_keys:
                sql = f"INSERT INTO '{meta_table}' (name_table, attribute, description) VALUES('{table_name}','{f}', '{d}')"
                cursor.execute(sql)
        self.conn.commit()