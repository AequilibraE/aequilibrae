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
        if not os.path.isfile(path_to_file):
            if not new_project:
                raise FileNotFoundError(
                    "Model does not exist. Check your path or use the new_project=True flag to create a new project"
                )
            else:
                shutil.copyfile(spatialite_database, path_to_file)
        self.path_to_file = path_to_file
        self.conn = sqlite3.connect(path_to_file)
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
            warn("AequilibraE might not work as intended without spatialite. {}".format(e.args))
        os.chdir(pth)

        # Now we populate all the stuff we want from this guy
        self.source = path_to_file
        self.network = Network(self)
