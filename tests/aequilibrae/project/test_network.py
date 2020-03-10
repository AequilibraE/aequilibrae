from unittest import TestCase
import sqlite3
from tempfile import gettempdir
import os
import platform
from functools import reduce
from aequilibrae.project import Project
from aequilibrae.project.network.network import Network
from aequilibrae.parameters import Parameters
from os.path import join, dirname
from warnings import warn
from random import random
from aequilibrae.project.spatialite_connection import spatialite_connection


class TestNetwork(TestCase):
    def setUp(self) -> None:
        self.file = os.path.join(gettempdir(), "aequilibrae_project_test.sqlite")
        self.project = Project()
        self.project.new(self.file)
        self.source = self.file
        self.file2 = os.path.join(gettempdir(), "aequilibrae_project_test2.sqlite")
        self.conn = sqlite3.connect(self.file2)
        self.conn = spatialite_connection(self.conn)
        self.network = Network(self)

    def tearDown(self) -> None:
        try:
            self.project.conn.close()
            os.unlink(self.file)

            self.conn.close()
            os.unlink(self.file2)
        except Exception as e:
            warn(f'Could not delete. {e.args}')

    def test_create_from_osm(self):
        thresh = 0.05
        if os.environ.get('GITHUB_WORKFLOW', 'ERROR') == 'Code coverage':
            thresh = 1.01

        if random() < thresh:
            # self.network.create_from_osm(west=153.1136245, south=-27.5095487, east=153.115, north=-27.5085, modes=["car"])
            self.project.network.create_from_osm(west=-112.185, south=36.59, east=-112.179, north=36.60)
            curr = self.project.conn.cursor()

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
        else:
            print('Skipped check to not load OSM servers')

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
            twf.extend([f"{k}_ab", f"{k}_ba"])

        for f in owf + twf:
            if f not in fields:
                self.fail(f"Field {f} not added to links table")

        curr = self.conn.cursor()
        curr.execute("""PRAGMA table_info(nodes);""")
        nfields = curr.fetchall()
        nfields = [x[1] for x in nfields]

        flds = reduce(lambda a, b: dict(a, **b), p["nodes"]["fields"])
        flds = list(flds.keys())

        for f in flds:
            if f not in nfields:
                self.fail(f"Field {f} not added to nodes table")

    # def test_count_links(self):
    #     self.fail()
    #
    # def test_count_nodes(self):
    #     self.fail()
