from unittest import TestCase
import sqlite3
from tempfile import gettempdir
import os
import shutil
import platform
from time import sleep
from functools import reduce
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
        # os.unlink(self.file)

    def test_create_from_osm(self):
        self.network.create_from_osm(west=-112.185, south=36.59, east=-112.179, north=36.60, modes=["car"])

        curr = self.conn.cursor()

        curr.execute("""select count(*) from links""")
        lks = curr.fetchone()

        curr.execute("""select count(distinct osm_id) from links""")
        osmids = curr.fetchone()

        if osmids >= lks:
            self.fail("OSM links not broken down properly")

        curr.execute("""select count(*) from nodes""")
        nds = curr.fetchone()

        if lks > nds:
            self.fail("We imported more links than nodes. Something wrong here")

    def test_create_empty_tables(self):
        self.network.create_empty_tables()
        p = Parameters().parameters["network"]

        curr = self.conn.cursor()
        curr.execute("""PRAGMA table_info(links);""")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        oneway = reduce(lambda a, b: dict(a, **b), p["links"]["fields"]["one-way"])
        owf = list(oneway.keys())
        twoway = reduce(lambda a, b: dict(a, **b), p["links"]["fields"]["two-way"])
        twf = []
        for k in list(twoway.keys()):
            twf.extend(["{}_ab".format(k), "{}_ba".format(k)])

        for f in owf + twf:
            if f not in fields:
                self.fail("Field {} not added to links table".format(f))

        curr = self.conn.cursor()
        curr.execute("""PRAGMA table_info(nodes);""")
        nfields = curr.fetchall()
        nfields = [x[1] for x in nfields]

        flds = reduce(lambda a, b: dict(a, **b), p["nodes"]["fields"])
        flds = list(flds.keys())

        for f in flds:
            if f not in nfields:
                self.fail("Field {} not added to nodes table".format(f))

    # def test_count_links(self):
    #     self.fail()
    #
    # def test_count_nodes(self):
    #     self.fail()
