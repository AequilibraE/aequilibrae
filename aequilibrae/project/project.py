import sqlite3
import os
import platform
import shutil
from warnings import warn
from aequilibrae.project.network.network import Network
from aequilibrae.parameters import Parameters
from aequilibrae.reference_files import spatialite_database


class Project:
    def __init__(self, path_to_file: str, new_project=False):
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
        self.get_spatialite_connection()
        self.network = Network(self)

    def get_spatialite_connection(self):
        self.conn.enable_load_extension(True)
        plat = platform.platform()
        pth = os.getcwd()
        if "WINDOWS" in plat.upper():
            par = Parameters()
            spatialite_path = par.parameters["system"]["spatialite_path"]
            if os.path.isfile(os.path.join(spatialite_path, "mod_spatialite.dll")):
                os.chdir(spatialite_path)
        try:
            self.conn.load_extension("mod_spatialite")
        except Exception as e:
            warn(f"AequilibraE might not work as intended without spatialite. {e.args}")
        os.chdir(pth)

    def __create_empty_project(self):
        shutil.copyfile(spatialite_database, self.path_to_file)
        self.conn = sqlite3.connect(self.path_to_file)
        self.__create_modes_table()

    def __create_modes_table(self):

        create_query = """CREATE TABLE 'modes' (mode_name VARCHAR NOT NULL,
                                                mode_id VARCHAR PRIMARY KEY UNIQUE,
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
