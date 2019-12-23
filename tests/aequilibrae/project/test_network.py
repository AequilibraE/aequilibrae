from unittest import TestCase
import sqlite3
from tempfile import gettempdir
import os
import shutil
import platform
from time import sleep
from aequilibrae.project.network.network import Network
from aequilibrae.parameters import Parameters
from aequilibrae.reference_files import spatialite_database


class TestNetwork(TestCase):
    def setUp(self) -> None:
        self.file = os.path.join(gettempdir(), "aequilibrae_project_test.sqlite")
        shutil.copyfile(spatialite_database, self.file)
        self.conn = sqlite3.connect(self.file)
        self.conn.enable_load_extension(True)
        plat = platform.platform()
        pth = os.getcwd()
        if "WINDOWS" in plat.upper():
            par = Parameters()
            spatialite_path = par.parameters["system"]["spatialite_path"]
            if os.path.isfile(os.path.join(spatialite_path, "mod_spatialite.dll")):
                os.chdir(spatialite_path)
        self.conn.load_extension("mod_spatialite")
        os.chdir(pth)

        self.network = Network(self)

    def tearDown(self) -> None:
        self.conn.close()
        os.unlink(self.file)

    def test_create_from_osm(self):
        self.network.create_from_osm(west=-112.185, south=36.59, east=-112.179, north=36.60, modes=["car"])
        i = 0
        while self.network.downloader.isRunning():
            sleep(1)
            i += 1
            print("{} seconds".format(i))
        sleep(1)
        print(self.network.json)

    def test_count_links(self):
        self.fail()

    def test_count_nodes(self):
        self.fail()
